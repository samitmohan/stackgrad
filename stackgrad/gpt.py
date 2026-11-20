import numpy as np

from .attention import Attention, KVCache
from .checkpoint import checkpoint
from .layers import DepthAttention, Embedding, Linear, RMSNorm, SiLUMLP, SwiGLU
from .module import Module
from .sampling import sample, sample_on_device
from .tensor import no_grad


def trunc_normal(shape, std):
  """N(0, std^2) resampled until every value lies within 3 std."""
  x = np.random.randn(*shape)
  while (bad := np.abs(x) > 3).any():
    x[bad] = np.random.randn(bad.sum())
  return (x * std).astype(np.float32)


class Block(Module):
  """norm, rope, and ffn set the CS336 ablations; the defaults are the modern block."""
  def __init__(self, d_model, n_heads, n_kv_heads, d_ff, rope_theta, head_dim=None, qk_norm=False, norm_eps=1e-5,
               norm="pre", rope=True, ffn="swiglu"):
    super().__init__()
    self.norm = norm
    self.attn_norm = None if norm == "none" else RMSNorm(d_model, norm_eps)
    self.attn = Attention(d_model, n_heads, n_kv_heads, rope_theta, head_dim, qk_norm, norm_eps, rope)
    self.ffn_norm = None if norm == "none" else RMSNorm(d_model, norm_eps)
    self.ffn = SwiGLU(d_model, d_ff) if ffn == "swiglu" else SiLUMLP(d_model, d_ff)

  def forward(self, x, cache=None, pad=None):
    if self.norm == "post":  # the original transformer's order
      x = self.attn_norm(x + self.attn(x, cache, pad))
      return self.ffn_norm(x + self.ffn(x))
    if self.norm == "none":
      x = x + self.attn(x, cache, pad)
      return x + self.ffn(x)
    x = x + self.attn(self.attn_norm(x), cache, pad)
    return x + self.ffn(self.ffn_norm(x))

  def sublayers(self, cache=None, pad=None):
    """Attention and feed-forward as functions of their input, without forward()'s residual adds."""
    return [lambda x: self.attn(self.attn_norm(x), cache, pad), lambda x: self.ffn(self.ffn_norm(x))]


class TransformerLM(Module):
  """Decoder-only transformer LM: the CS336 model by default, Qwen3 with the options qwen3.py sets."""
  def __init__(self, vocab_size, context_length, d_model, n_layers, n_heads, d_ff,
               n_kv_heads=None, rope_theta=10000.0, head_dim=None, qk_norm=False, norm_eps=1e-5, tie_embeddings=False,
               norm="pre", rope=True, ffn="swiglu", attn_res=False, attn_res_block=8):
    super().__init__()
    assert not attn_res or norm == "pre", "Attention Residuals is defined on pre-norm blocks"
    self.context_length = context_length
    self.attn_res_block = attn_res_block
    self.depth = [DepthAttention(d_model, norm_eps) for _ in range(2 * n_layers + 1)] if attn_res else None
    self.embed = Embedding(vocab_size, d_model)
    self.blocks = [Block(d_model, n_heads, n_kv_heads, d_ff, rope_theta, head_dim, qk_norm, norm_eps, norm, rope, ffn)
                   for _ in range(n_layers)]
    self.norm = None if norm == "none" else RMSNorm(d_model, norm_eps)  # "none" removes every RMSNorm, the final one too
    self.head = None if tie_embeddings else Linear(d_model, vocab_size, bias=False)
    self.checkpoint_blocks = False  # recompute each block in backward instead of storing its activations

    # CS336 init
    for p in self.parameters():
      if p.ndim == 2:
        p.data = trunc_normal(p.shape, np.sqrt(2.0 / sum(p.shape)))
    self.embed.weight.data = trunc_normal(self.embed.weight.shape, 1.0)

  def forward(self, idx, caches=None, pad=None):
    """idx (B, T) ints -> logits (B, T, vocab); caches is one KVCache per block, pad (B,) the left padding."""
    return self.project(self.hidden(idx, caches, pad))

  def hidden(self, idx, caches=None, pad=None):
    """Final normalized hidden states (B, T, d_model): the model without the output projection."""
    x = self.embed(idx)
    if self.depth is not None:
      x = self.attn_res(x, caches, pad)
      return x if self.norm is None else self.norm(x)
    for i, block in enumerate(self.blocks):
      if self.checkpoint_blocks and caches is None and pad is None:
        x = checkpoint(block, x)
      else:
        x = block(x, None if caches is None else caches[i], pad)
    return x if self.norm is None else self.norm(x)

  def attn_res(self, x, caches=None, pad=None):
    """Block Attention Residuals over the embedding x: the hidden states before the final norm."""
    # shortcut: no activation checkpointing here; add it per block if attn_res ever trains a model too big to fit
    blocks, partial = [x], None  # finished block sums (the embedding first), and the running sum of this block
    layers = [f for i, b in enumerate(self.blocks) for f in b.sublayers(None if caches is None else caches[i], pad)]
    for i, f in enumerate(layers):
      out = f(self.depth[i](blocks if partial is None else blocks + [partial]))
      partial = out if partial is None else partial + out
      if (i + 1) % self.attn_res_block == 0:
        blocks, partial = blocks + [partial], None
    return self.depth[-1](blocks if partial is None else blocks + [partial])

  def project(self, h):
    """Hidden states -> logits, through the head or the tied embedding table."""
    return self.head(h) if self.head is not None else h @ self.embed.weight.T

  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_id=None, pad=None):
    """Extends idx (B, T0) by up to max_new_tokens until every row emits eos_id; pad (B,) is left padding."""
    idx = np.asarray(idx)
    assert idx.shape[1] + max_new_tokens <= self.context_length, "prompt + new tokens exceed context_length"
    caches = [KVCache() for _ in self.blocks]
    done = np.zeros(len(idx), dtype=bool)
    with no_grad():
      logits = self.project(self.hidden(idx, caches, pad)[:, -1])
      for _ in range(max_new_tokens):
        if top_k is None and top_p is None:
          next_id = sample_on_device(logits, temperature)
        else:
          next_id = sample(logits.numpy(), temperature, top_k, top_p)
        idx = np.concatenate([idx, next_id[:, None]], axis=1)
        if eos_id is not None:
          done |= next_id == eos_id
          if done.all():
            break
        logits = self.project(self.hidden(next_id[:, None], caches, pad)[:, -1])
    return idx
