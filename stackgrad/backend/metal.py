import mlx.core as mx
import numpy as np
from mlx.core import (
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
    sigmoid,
    sign,
    stack,
    triu,
    where,
    zeros_like,
)

from ..tensor import Tensor
from . import cpu, metal_attention, metal_quant

__all__ = [
    "asarray",
    "broadcast_to",
    "capture",
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
    "im2col",
    "inf",
    "is_float",
    "log",
    "ones_like",
    "pad",
    "quant_matmul",
    "realize",
    "repeat",
    "scatter_add",
    "sigmoid",
    "sign",
    "stack",
    "to_numpy",
    "triu",
    "where",
    "write",
    "zeros_like",
]


def asarray(x, dtype=None):
  if isinstance(x, mx.array):
    return x if dtype is None or x.dtype == dtype else x.astype(dtype)
  return mx.array(x) if dtype is None else mx.array(x, dtype=dtype)


def to_numpy(x):
  # numpy has no bfloat16; widening to float32 is exact
  return np.array(x.astype(mx.float32) if x.dtype == mx.bfloat16 else x)


def dtype(name):
  return getattr(mx, name)


def is_float(dt):
  return mx.issubdtype(dt, mx.floating)


realize = mx.eval
clear_cache = mx.clear_cache  # mlx keeps freed buffers cached for reuse until this
mx.set_cache_limit(1 << 30)  # freed buffers mlx keeps for reuse: 8-chat serving held 2.4 GB of them uncapped, 1.1 GB capped, same speed
fuse = mx.compile  # traces fn once per input shape and fuses its elementwise ops into shared kernels


# fn takes and returns Tensors; mx.compile runs its Python once per input shape, and later calls replay one fused graph
def capture(fn):
  traced = mx.compile(lambda *xs: [t.data for t in fn(*(Tensor(x, device="metal") for x in xs))])
  return lambda *ts: [Tensor(y, device="metal") for y in traced(*(t.data for t in ts))]


def scatter_add(shape, key, values):
  return mx.zeros(shape, dtype=values.dtype).at[key].add(values)


# shortcut: col2im (conv backward) round-trips through the CPU, port it to MLX if a CNN trains on metal
def im2col(x, *args):
  return cpu.im2col(x, *args, xp=mx)


def col2im(cols, *args):
  return mx.array(cpu.col2im(to_numpy(cols), *args))


def write(buf, x, pos):
  """buf with x written along axis 2 from pos (an int, or an array inside a compiled trace), in place when nothing else holds buf."""
  return mx.slice_update(buf, x, asarray(pos).astype(mx.int32).reshape(1), axes=(2,))


def fused_attention(q, k, v, n):
  """Metal attention forward over the first n keys, or None for shapes the kernels don't cover."""
  D = q.shape[-1]
  if D % 8 or D > 128:
    return None
  # one query row (decode) leaves v3's 32-row threadgroups mostly idle; v1 measured faster there
  return metal_attention.attention(q, k, v, "v1" if q.shape[2] == 1 else "v3", asarray(n).reshape(1))


def quant_matmul(x, packed, scales, biases, bits, group):
  """x (M, K) @ dequantized W (N, K).T as one Metal kernel, for decode-sized M."""
  return metal_quant.quant_matmul(x, packed, scales, biases, bits, group)
