import numpy as np

from . import backend
from .layers import Linear
from .module import Module
from .tensor import Tensor


def quantize(w, bits=4, group=64):
    """w (out, in) -> packed uint32 (out, in * bits / 32), scales, biases (out, in / group), in MLX's layout."""
    n_out, n_in = w.shape
    assert 32 % bits == 0 and n_in % group == 0 and group % (32 // bits) == 0, (w.shape, bits, group)
    g = w.astype(np.float32).reshape(n_out, n_in // group, group)
    lo, hi = g.min(axis=-1, keepdims=True), g.max(axis=-1, keepdims=True)
    scale = np.where(hi > lo, (hi - lo) / (2**bits - 1), 1.0)  # a constant group: any scale works, bias carries it
    q = np.clip(np.round((g - lo) / scale), 0, 2**bits - 1).astype(np.uint32).reshape(n_out, -1, 32 // bits)
    shifts = np.arange(32 // bits, dtype=np.uint32) * bits
    packed = np.bitwise_or.reduce(q << shifts, axis=-1)
    return packed, scale[..., 0], lo[..., 0]


def dequantize(packed, scales, biases, bits, group, xp):
    """The inverse of quantize, in the backend `xp` and in the scales' dtype."""
    shifts = xp.asarray(np.arange(32 // bits, dtype=np.uint32) * bits)
    q = (packed[..., None] >> shifts) & (2**bits - 1)
    q = q.reshape(packed.shape[0], -1, group).astype(scales.dtype)
    return (q * scales[..., None] + biases[..., None]).reshape(packed.shape[0], -1)


class QuantLinear(Module):
    """Inference-only, bias-free Linear with group-quantized weights; build it after moving the model."""
    def __init__(self, weight, bits, group):
        super().__init__()
        xp = backend.get(weight.device)
        packed, scales, biases = quantize(weight.numpy().T, bits, group)  # stackgrad keeps (in, out)
        self.device, self.bits, self.group = weight.device, bits, group
        self.packed = xp.asarray(packed)
        self.scales, self.biases = xp.asarray(scales, dtype=weight.dtype), xp.asarray(biases, dtype=weight.dtype)

    def forward(self, x):
        assert not x.requires_grad, "QuantLinear is inference only: run it under no_grad()"
        assert x.device == self.device, f"input on {x.device}, weights on {self.device}"
        xp = backend.get(self.device)
        flat = x.data.reshape(-1, x.shape[-1])
        fused = getattr(xp, "quant_matmul", None)
        if fused is not None and flat.shape[0] <= 8:  # decode: read the packed weights directly
            y = fused(flat, self.packed, self.scales, self.biases, self.bits, self.group)
        else:  # prefill: dequantize once, then one large matmul
            y = flat @ dequantize(self.packed, self.scales, self.biases, self.bits, self.group, xp).T
        return Tensor(y.reshape(*x.shape[:-1], y.shape[-1]), device=self.device)

    def nbytes(self):
        return sum(a.nbytes for a in (self.packed, self.scales, self.biases))


def quantize_model(model, bits=4, group=64):
    """Replaces every TransformerLM matmul, head included, with a QuantLinear; a tied head gets its own copy."""
    for block in model.blocks:
        for parent in (block.attn, block.ffn):
            for name, val in list(parent.__dict__.items()):
                if isinstance(val, Linear):
                    assert val.bias is None, f"{name} has a bias"
                    setattr(parent, name, QuantLinear(val.weight, bits, group))
    head = model.head.weight if model.head is not None else model.embed.weight.T
    model.head = QuantLinear(head, bits, group)
    return model
