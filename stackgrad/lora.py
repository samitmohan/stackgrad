import numpy as np

from .layers import Linear
from .module import Module
from .tensor import Tensor

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3")  # every matmul in a TransformerLM block


class LoRALinear(Module):
    """A frozen Linear plus a trained low-rank update: y = x W + (alpha / r) (x A) B, B starting at zero."""
    def __init__(self, base, r, alpha):
        super().__init__()
        n_in, n_out = base.weight.shape
        self.base = base
        # cast after dividing: numpy 2 promotes float32 / np.sqrt's float64 scalar to float64
        init = (np.random.randn(n_in, r) / np.sqrt(n_in)).astype(np.float32)
        self.A = Tensor(init, requires_grad=True, device=base.weight.device)
        self.B = Tensor(np.zeros((r, n_out), np.float32), requires_grad=True, device=base.weight.device)
        self.scale = alpha / r

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        delta = (flat @ self.A * self.scale) @ self.B  # scaling the rank-r middle, not the full-width output, saves a copy
        return self.base(x) + delta.reshape(*x.shape[:-1], delta.shape[-1])


def apply_lora(model, r=16, alpha=32):
    """Freezes a TransformerLM and wraps each target Linear in a LoRALinear, so only A and B train."""
    for _, p in model.named_parameters():
        p.requires_grad = False
    for block in model.blocks:
        for parent in (block.attn, block.ffn):
            for name in TARGETS:
                if isinstance(getattr(parent, name, None), Linear):
                    setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha))
    return model


def save_adapter(model, path, r, alpha):
    np.savez(path, r=r, alpha=alpha, **{name: p.numpy() for name, p in model.named_parameters()})


def merge_adapter(model, path):
    """Add a saved adapter's (alpha / r) A B into the matching weights of a plain (un-wrapped) model."""
    adapter = np.load(path)
    scale = float(adapter["alpha"]) / float(adapter["r"])
    params = dict(model.named_parameters())
    layers = [k[:-len(".A")] for k in adapter.files if k.endswith(".A")]
    for layer in layers:
        w = params[f"{layer}.weight"]  # a KeyError here means the adapter was trained on another model
        assert w.device == "cpu", "merge the adapter before moving the model to a device"
        w.data = w.data + (scale * (adapter[f"{layer}.A"] @ adapter[f"{layer}.B"])).astype(w.dtype)
    return len(layers)
