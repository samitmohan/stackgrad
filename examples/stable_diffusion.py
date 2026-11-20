import argparse
import itertools
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import regex

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stackgrad import Tensor, backend, cat, no_grad, where
from stackgrad.attention import scaled_dot_product_attention
from stackgrad.conv import Conv2d
from stackgrad.functional import silu
from stackgrad.layers import Dropout, Embedding, GroupNorm, LayerNorm, Linear
from stackgrad.module import Module
from stackgrad.state import iter_safetensors
from stackgrad.tokenizer import bytes_to_unicode

LATENT_SCALE = 0.18215  # the VAE's scaling_factor: latents have about unit variance after it


class ClipTokenizer:
    """CLIP's byte-level BPE: unlike GPT-2's it lowercases, and a word's last piece carries a </w> marker."""
    PATTERN = regex.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+")

    def __init__(self, path):
        self.vocab = json.loads((Path(path) / "vocab.json").read_text())
        lines = (Path(path) / "merges.txt").read_text().split("\n")[1:]  # line 0 is a version header
        self.ranks = {tuple(line.split()): i for i, line in enumerate(lines) if line}
        self.byte_char = bytes_to_unicode()
        self.bos, self.eos = self.vocab["<|startoftext|>"], self.vocab["<|endoftext|>"]

    def bpe(self, word):
        parts = [self.byte_char[b] for b in word.encode()]
        parts[-1] += "</w>"
        while len(parts) > 1:
            rank, i = min((self.ranks.get(pair, math.inf), i) for i, pair in enumerate(itertools.pairwise(parts)))
            if rank == math.inf:
                break
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        return parts

    def encode(self, text, length=77):
        # shortcut: no ftfy or HTML unescaping like the reference tokenizer, so plain-text prompts only; add both for scraped text
        words = self.PATTERN.findall(" ".join(text.split()).lower())
        ids = [self.vocab[piece] for w in words for piece in self.bpe(w)][:length - 2]
        return [self.bos, *ids, self.eos] + [self.eos] * (length - 2 - len(ids))


def attend(q, k, v, heads, causal=False):
    """One head at a time: a guided UNet call peaked at 6.04 GB this way, 7.82 GB with all 8 heads' scores (M4)."""
    B, T, C = q.shape

    def split(x):
        return x.reshape(B, x.shape[1], heads, C // heads).transpose(1, 2)
    q, k, v = split(q), split(k), split(v)
    out = cat([scaled_dot_product_attention(q[:, h:h + 1], k[:, h:h + 1], v[:, h:h + 1], causal=causal) for h in range(heads)], axis=1)
    return out.transpose(1, 2).reshape(B, T, C)


def quick_gelu(x):
    return x * (x * 1.702).sigmoid()


class ClipAttention(Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (Linear(d, d) for _ in range(4))

    def forward(self, x):
        return self.out_proj(attend(self.q_proj(x), self.k_proj(x), self.v_proj(x), self.heads, causal=True))


class ClipMLP(Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.fc1, self.fc2 = Linear(d, d_ff), Linear(d_ff, d)

    def forward(self, x):
        return self.fc2(quick_gelu(self.fc1(x)))


class ClipLayer(Module):
    def __init__(self, d, heads, d_ff):
        super().__init__()
        self.layer_norm1, self.self_attn = LayerNorm(d), ClipAttention(d, heads)
        self.layer_norm2, self.mlp = LayerNorm(d), ClipMLP(d, d_ff)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class ClipEmbeddings(Module):
    def __init__(self, vocab, positions, d):
        super().__init__()
        self.token_embedding, self.position_embedding = Embedding(vocab, d), Embedding(positions, d)


class ClipEncoder(Module):
    def __init__(self, d, heads, d_ff, n_layers):
        super().__init__()
        self.layers = [ClipLayer(d, heads, d_ff) for _ in range(n_layers)]


class ClipTextModel(Module):
    """CLIP ViT-L/14's text tower. Stable Diffusion reads all 77 of its final vectors."""
    def __init__(self, vocab=49408, positions=77, d=768, heads=12, d_ff=3072, n_layers=12):
        super().__init__()
        self.embeddings = ClipEmbeddings(vocab, positions, d)
        self.encoder = ClipEncoder(d, heads, d_ff, n_layers)
        self.final_layer_norm = LayerNorm(d)

    def forward(self, ids):
        x = self.embeddings.token_embedding(ids) + self.embeddings.position_embedding(np.arange(ids.shape[1]))
        for layer in self.encoder.layers:
            x = layer(x)
        return self.final_layer_norm(x)


def gelu(x):
    """Erf GELU like diffusers, not tanh (5e-4 off); backends lack erf, so A&S 7.1.26 (error < 1.5e-7)."""
    z = x * (1 / math.sqrt(2))
    a = where(z > 0, z, -z)
    t = (a * 0.3275911 + 1.0).recip()
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf_a = 1.0 - poly * (-(a * a)).exp()
    return x * 0.5 * (where(z > 0, erf_a, -erf_a) + 1.0)


class ResnetBlock(Module):
    def __init__(self, c_in, c_out, temb_dim=None, eps=1e-5):
        super().__init__()
        self.norm1, self.conv1 = GroupNorm(32, c_in, eps), Conv2d(c_in, c_out, 3, padding=1)
        self.time_emb_proj = Linear(temb_dim, c_out) if temb_dim else None
        self.norm2, self.conv2 = GroupNorm(32, c_out, eps), Conv2d(c_out, c_out, 3, padding=1)
        self.conv_shortcut = Conv2d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x, temb=None):
        h = self.conv1(silu(self.norm1(x)))
        if temb is not None:
            h = h + self.time_emb_proj(silu(temb)).reshape(x.shape[0], -1, 1, 1)
        h = self.conv2(silu(self.norm2(h)))
        return (x if self.conv_shortcut is None else self.conv_shortcut(x)) + h


class CrossAttention(Module):
    def __init__(self, c, context_dim, heads, bias=False):
        super().__init__()
        self.heads = heads
        self.to_q = Linear(c, c, bias=bias)
        self.to_k, self.to_v = Linear(context_dim, c, bias=bias), Linear(context_dim, c, bias=bias)
        self.to_out = [Linear(c, c)]

    def forward(self, x, context=None):
        context = x if context is None else context
        return self.to_out[0](attend(self.to_q(x), self.to_k(context), self.to_v(context), self.heads))


class GEGLU(Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = Linear(c_in, 2 * c_out)

    def forward(self, x):
        h = self.proj(x)
        n = h.shape[-1] // 2
        return h[..., :n] * gelu(h[..., n:])


class FeedForward(Module):
    def __init__(self, c):
        super().__init__()
        self.net = [GEGLU(c, 4 * c), Dropout(0.0), Linear(4 * c, c)]  # the Dropout keeps the checkpoint's numbering

    def forward(self, x):
        return self.net[2](self.net[0](x))


class TransformerBlock(Module):
    def __init__(self, c, context_dim, heads):
        super().__init__()
        self.norm1, self.attn1 = LayerNorm(c), CrossAttention(c, c, heads)
        self.norm2, self.attn2 = LayerNorm(c), CrossAttention(c, context_dim, heads)
        self.norm3, self.ff = LayerNorm(c), FeedForward(c)

    def forward(self, x, context):
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        return x + self.ff(self.norm3(x))


class Transformer2D(Module):
    def __init__(self, c, context_dim=768, heads=8):
        super().__init__()
        self.norm = GroupNorm(32, c, eps=1e-6)
        self.proj_in, self.proj_out = Conv2d(c, c, 1), Conv2d(c, c, 1)
        self.transformer_blocks = [TransformerBlock(c, context_dim, heads)]

    def forward(self, x, context):
        B, C, H, W = x.shape
        h = self.proj_in(self.norm(x)).reshape(B, C, H * W).transpose(1, 2)
        for block in self.transformer_blocks:
            h = block(h, context)
        return x + self.proj_out(h.transpose(1, 2).reshape(B, C, H, W))


class Downsample(Module):
    def __init__(self, c):
        super().__init__()
        self.conv = Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(Module):
    """Nearest-neighbor 2x, then a 3x3 convolution."""
    def __init__(self, c):
        super().__init__()
        self.conv = Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        return self.conv(x.reshape(B, C, H, 1, W, 1).expand(B, C, H, 2, W, 2).reshape(B, C, 2 * H, 2 * W))


class DownBlock(Module):
    def __init__(self, c_in, c_out, attention, downsample):
        super().__init__()
        self.resnets = [ResnetBlock(c_in if i == 0 else c_out, c_out, 1280) for i in range(2)]
        self.attentions = [Transformer2D(c_out) for _ in range(2)] if attention else []
        self.downsamplers = [Downsample(c_out)] if downsample else []

    def forward(self, x, temb, context, skips):
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, temb)
            if self.attentions:
                x = self.attentions[i](x, context)
            skips.append(x)
        for down in self.downsamplers:
            x = down(x)
            skips.append(x)
        return x


class UpBlock(Module):
    def __init__(self, c_ins, c_out, attention, upsample):
        super().__init__()
        self.resnets = [ResnetBlock(c_in, c_out, 1280) for c_in in c_ins]
        self.attentions = [Transformer2D(c_out) for _ in c_ins] if attention else []
        self.upsamplers = [Upsample(c_out)] if upsample else []

    def forward(self, x, temb, context, skips):
        for i, resnet in enumerate(self.resnets):
            x = resnet(cat([x, skips.pop()], axis=1), temb)
            if self.attentions:
                x = self.attentions[i](x, context)
        for up in self.upsamplers:
            x = up(x)
        return x


class MidBlock(Module):
    def __init__(self, resnets, attention):
        super().__init__()
        self.resnets, self.attentions = resnets, [attention]

    def forward(self, x, temb=None, context=None):
        return self.resnets[1](self.attentions[0](self.resnets[0](x, temb), context), temb)


class TimestepEmbedding(Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.linear_1, self.linear_2 = Linear(c_in, c_out), Linear(c_out, c_out)

    def forward(self, x):
        return self.linear_2(silu(self.linear_1(x)))


def timestep_features(t, batch, dim=320):
    angles = t * np.exp(-math.log(10000) * np.arange(dim // 2) / (dim // 2))
    return np.tile(np.concatenate([np.cos(angles), np.sin(angles)]), (batch, 1)).astype(np.float32)


class UNet(Module):
    CHANNELS = (320, 640, 1280, 1280)

    def __init__(self):
        super().__init__()
        self.time_embedding = TimestepEmbedding(320, 1280)
        self.conv_in = Conv2d(4, 320, 3, padding=1)
        skips, c = [320], 320  # channels of each skip the down path pushes, to size the up path's inputs
        self.down_blocks = []
        for i, c_out in enumerate(self.CHANNELS):
            last = i == len(self.CHANNELS) - 1
            self.down_blocks.append(DownBlock(c, c_out, attention=not last, downsample=not last))
            skips += [c_out] * (2 if last else 3)
            c = c_out
        self.mid_block = MidBlock([ResnetBlock(c, c, 1280) for _ in range(2)], Transformer2D(c))
        self.up_blocks = []
        for i, c_out in enumerate(reversed(self.CHANNELS)):
            c_ins = []
            for _ in range(3):
                c_ins.append(c + skips.pop())
                c = c_out
            self.up_blocks.append(UpBlock(c_ins, c_out, attention=i > 0, upsample=i < len(self.CHANNELS) - 1))
        self.conv_norm_out, self.conv_out = GroupNorm(32, 320), Conv2d(320, 4, 3, padding=1)

    def forward(self, x, t, context):
        temb = self.time_embedding(Tensor(timestep_features(t, x.shape[0]), device=x.device))
        x = self.conv_in(x)
        skips = [x]
        for block in self.down_blocks:
            x = block(x, temb, context, skips)
        x = self.mid_block(x, temb, context)
        for block in self.up_blocks:
            x = block(x, temb, context, skips)
        return self.conv_out(silu(self.conv_norm_out(x)))


class SpatialSelfAttention(CrossAttention):
    def __init__(self, c):
        super().__init__(c, c, heads=1, bias=True)
        self.group_norm = GroupNorm(32, c, eps=1e-6)

    def forward(self, x, context=None):  # context: unused, the VAE attends within the image
        B, C, H, W = x.shape
        h = super().forward(self.group_norm(x).reshape(B, C, H * W).transpose(1, 2))
        return x + h.transpose(1, 2).reshape(B, C, H, W)


class DecoderBlock(Module):
    def __init__(self, c_in, c_out, upsample):
        super().__init__()
        self.resnets = [ResnetBlock(c_in if i == 0 else c_out, c_out, eps=1e-6) for i in range(3)]
        self.upsamplers = [Upsample(c_out)] if upsample else []

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        for up in self.upsamplers:
            x = up(x)
        return x


class Decoder(Module):
    CHANNELS = (512, 512, 256, 128)

    def __init__(self):
        super().__init__()
        self.conv_in = Conv2d(4, 512, 3, padding=1)
        self.mid_block = MidBlock([ResnetBlock(512, 512, eps=1e-6) for _ in range(2)], SpatialSelfAttention(512))
        c, self.up_blocks = 512, []
        for i, c_out in enumerate(self.CHANNELS):
            self.up_blocks.append(DecoderBlock(c, c_out, upsample=i < len(self.CHANNELS) - 1))
            c = c_out
        self.conv_norm_out, self.conv_out = GroupNorm(32, 128, eps=1e-6), Conv2d(128, 3, 3, padding=1)

    def forward(self, z):
        x = self.mid_block(self.conv_in(z))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(silu(self.conv_norm_out(x)))


class VAE(Module):
    def __init__(self):
        super().__init__()
        self.post_quant_conv = Conv2d(4, 4, 1)
        self.decoder = Decoder()

    def decode(self, latents):
        return self.decoder(self.post_quant_conv(latents * (1 / LATENT_SCALE)))


def load(model, path, prefix="", unused=()):
    params = dict(model.named_parameters())
    for name, w in iter_safetensors(path):
        name = name.removeprefix(prefix)
        if name.startswith(unused):
            continue
        if name not in params:  # LayerNorm calls its scale and shift gamma and beta
            name = re.sub(r"\.bias$", ".beta", re.sub(r"\.weight$", ".gamma", name))
        assert name in params, f"checkpoint tensor {name} has no place in the model"
        p = params.pop(name)
        w = w.astype(np.float32)
        if w.ndim == 4:  # Conv2d: (out, in, kh, kw) -> (out, in*kh*kw), im2col's order
            w = w.reshape(w.shape[0], -1)
        elif w.ndim == 2 and not name.endswith("embedding.weight"):  # Linear: (out, in) -> (in, out)
            w = np.ascontiguousarray(w.T)
        assert w.shape == p.shape, (name, w.shape, p.shape)
        p.data = w
    assert not params, f"missing from the checkpoint: {sorted(params)[:5]}"
    return model


def load_clip(path, device):
    return load(ClipTextModel(), Path(path) / "text_encoder/model.fp16.safetensors", "text_model.",
                unused=("embeddings.position_ids",)).to(device)


def load_unet(path, device):
    return load(UNet(), Path(path) / "unet/diffusion_pytorch_model.fp16.safetensors").to(device)


def load_vae(path, device):
    return load(VAE(), Path(path) / "vae/diffusion_pytorch_model.fp16.safetensors", unused=("encoder.", "quant_conv.")).to(device)


# SD 1.5's scaled_linear schedule: sqrt(beta) linear from sqrt(0.00085) to sqrt(0.012) over 1000 steps
ALPHAS_CUMPROD = np.cumprod(1 - np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=np.float32) ** 2)


def ddim_timesteps(steps):
    """Evenly spaced training timesteps, shifted by one (the scheduler's steps_offset)."""
    return [i * (1000 // steps) + 1 for i in reversed(range(steps))]


def ddim_step(x, noise, t, t_prev):
    a =ALPHAS_CUMPROD[t]
    a_prev = ALPHAS_CUMPROD[t_prev] if t_prev >= 0 else ALPHAS_CUMPROD[0]
    x0 = (x - np.sqrt(1 - a) * noise) / np.sqrt(a)
    return np.sqrt(a_prev) * x0 + np.sqrt(1 - a_prev) * noise


def denoise(unet, context, latents, steps, guidance, device):
    """context: (2, 77, 768), the empty prompt then the real one."""
    times = ddim_timesteps(steps)
    seconds = []
    with no_grad():
        for t, t_prev in zip(times, [*times[1:], -1]):
            start = time.perf_counter()
            noise = unet(Tensor(np.concatenate([latents, latents]), device=device), t, context).numpy()  # .numpy() waits for the GPU
            noise = noise[:1] + guidance * (noise[1:] - noise[:1])  # classifier-free guidance
            latents = ddim_step(latents, noise, t, t_prev)
            seconds.append(time.perf_counter() - start)
            print(f"step {len(seconds):3d}/{steps}  t={t:4d}  {seconds[-1]:.2f} s", flush=True)
    return latents, seconds


def generate(path, prompt, noise, steps, guidance, device):
    if device == "metal":
        import mlx.core as mx
        # MLX caches freed buffers: the VAE's many sizes took decoding to a 12.4 GB footprint, 6.0 GB uncached, same speed
        mx.set_cache_limit(0)
    tok = ClipTokenizer(Path(path) / "tokenizer")
    clip = load_clip(path, device)
    with no_grad():
        context = clip(np.array([tok.encode(""), tok.encode(prompt)])).numpy()
    del clip  # one network in memory at a time
    backend.get(device).clear_cache()
    unet = load_unet(path, device)
    latents, seconds = denoise(unet, Tensor(context, device=device), noise, steps, guidance, device)
    del unet
    backend.get(device).clear_cache()  # the VAE's buffer shapes can't reuse the UNet's ~6 GB
    vae = load_vae(path, device)
    with no_grad():
        image = vae.decode(Tensor(latents, device=device)).numpy()[0]
    return np.clip(image.transpose(1, 2, 0) / 2 + 0.5, 0, 1), seconds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/sd15")
    p.add_argument("--device", default="metal", choices=["cpu", "metal", "cuda"])
    p.add_argument("--prompt", default="a watercolor painting of a lighthouse on a cliff at dusk")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="bench/results/stable_diffusion.png")
    args = p.parse_args()

    from PIL import Image  # only to write the PNG

    noise = np.random.default_rng(args.seed).standard_normal((1, 4, args.size // 8, args.size // 8)).astype(np.float32)
    image, seconds = generate(args.model, args.prompt, noise, args.steps, args.guidance, args.device)
    Image.fromarray((image * 255).round().astype(np.uint8)).save(args.out)
    print(f"{args.out}: median {np.median(seconds):.2f} s per step on {args.device}")


if __name__ == "__main__":
    main()
