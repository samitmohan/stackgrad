import cupy as cp
import cupyx
from cupy import (
    broadcast_to,
    concatenate,
    cumsum,
    exp,
    expand_dims,
    float32,
    full,
    inf,
    log,
    ones_like,
    pad,
    repeat,
    sign,
    stack,
    triu,
    where,
    zeros_like,
)

from . import cpu, cuda_attention
from .cpu import dtype, fuse, is_float, realize  # shortcut: fuse runs each op on its own, try cupy.fuse or CUDA graphs on bqa3

__all__ = [
    "asarray",
    "broadcast_to",
    "clear_cache",
    "col2im",
    "concatenate",
    "cumsum",
    "dtype",
    "exp",
    "expand_dims",
    "float32",
    "full",
    "fuse",
    "fused_attention",
    "fused_attention_backward",
    "im2col",
    "inf",
    "is_float",
    "log",
    "ones_like",
    "pad",
    "realize",
    "repeat",
    "scatter_add",
    "sigmoid",
    "sign",
    "stack",
    "to_numpy",
    "triu",
    "where",
    "zeros_like",
]


def clear_cache():
  cp.get_default_memory_pool().free_all_blocks()

def asarray(x, dtype=None):
  return cp.asarray(x, dtype=dtype)


def to_numpy(x):
  return cp.asnumpy(x)


def fused_attention(q, k, v, n):
  """Triton FlashAttention forward over the first n keys, or None for shapes the kernels don't cover."""
  return cuda_attention.attention(q, k[:, :, :n], v[:, :, :n]) if cuda_attention.supported(q) else None


def fused_attention_backward(q, k, v, out, lse, grad):
  return cuda_attention.attention_backward(q, k, v, out, lse, grad)


def sigmoid(x):
  return 1.0 / (1.0 + cp.exp(-x))


def scatter_add(shape, key, values):
  out = cp.zeros(shape, dtype=values.dtype)
  cupyx.scatter_add(out, key, values)
  return out


# shortcut: col2im (conv backward) round-trips through the CPU, port it to CuPy if a CNN trains on cuda
def im2col(x, *args):
  return cpu.im2col(x, *args, xp=cp)


def col2im(cols, *args):
  return cp.asarray(cpu.col2im(cp.asnumpy(cols), *args))

