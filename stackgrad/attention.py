import numpy as np

from . import backend, mlops
from .layers import Linear, RMSNorm
from .module import Module
from .tensor import Tensor, _apply, cat


def rope_tables(start, T, D, theta):
  """cos and sin (B or 1, 1, T, D/2) of the angles at positions start .. start + T - 1, in Llama/Qwen's half-split layout."""
  half = D // 2
  positions = np.reshape(start, (-1, 1, 1)) + np.arange(T)[:, None]  # (B or 1, T, 1)
  angles = (positions * theta ** (-np.arange(half) / half))[:, None]  # (B or 1, 1, T, D/2)
  return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def apply_rope(x, cos, sin):
  """Rotates each (x[i], x[i + D/2]) pair by its position's angle, from rope_tables."""
  half = x.shape[-1] // 2
  x1, x2 = x[..., :half], x[..., half:]
  return cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def scaled_dot_product_attention(q, k, v, pad=None, causal=True, n=None):
  """q (B, H, T, D) over the first n positions of k, v (B, Hkv, S, D), all S if n is None, queries at the last T; pad (B,) masks left padding."""
  key_bias = None
  if pad is not None:
    # -1e9, not -inf: a padding query sees only padding, and an all -inf softmax is NaN that leaks via 0 * NaN
    key_bias = np.where(np.arange(k.shape[2]) < np.asarray(pad)[:, None], -1e9, 0.0).astype(np.float32)
  return _apply(mlops.Attention, q, k, v, key_bias=key_bias, causal=causal, n=n)


class KVCache:
  """Keys and values of earlier positions, written in place into buffers that grow 256 positions at a time, so decode steps keep their shapes."""
  def __init__(self, k=None, v=None, n=0, rope=None):
    # n counts the positions written: an int, or an array inside the compiled decode step, which brings its (cos, sin) as rope;
    # trace is that step, kept on a model's first cache
    self.k, self.v, self.n, self.rope, self.trace = k, v, n, rope, None

  # shortcut: growing copies the buffers and recompiles the decode trace (~20 ms, measured), once per 256 positions; size them up
  # front if long generations show it
  def reserve(self, T):
    """Grows the buffers to fit T more positions."""
    if self.n + T > self.k.shape[2]:
      grow = ((0, 0), (0, 0), (0, -(-(self.n + T) // 256) * 256 - self.k.shape[2]), (0, 0))
      self.k, self.v = (Tensor(backend.get(t.device).pad(t.data, grow), device=t.device) for t in (self.k, self.v))

  def update(self, k, v):
    """Writes k, v (B, Hkv, T, D) at position n; returns the whole buffers, valid up to the new n."""
    if self.k is None:
      self.k, self.v = k[:, :, :0], v[:, :, :0]  # empty, so reserve() allocates
    if self.rope is None:  # inside the compiled step n is an array, and the caller made room
      self.reserve(k.shape[2])
    self.k, self.v = _apply(mlops.Write, self.k, k, pos=self.n), _apply(mlops.Write, self.v, v, pos=self.n)
    self.n = self.n + k.shape[2]
    return self.k, self.v


class Attention(Module):
  """Multi-head attention with RoPE and grouped-query heads; qk_norm adds Qwen3's per-head RMSNorm on q, k."""
  def __init__(self, d_model, n_heads, n_kv_heads=None, rope_theta=10000.0, head_dim=None, qk_norm=False, norm_eps=1e-5,
               rope=True):
    super().__init__()
    self.rope = rope  # False is the NoPE ablation: causal masking alone orders tokens
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads or n_heads
    self.head_dim = head_dim or d_model // n_heads
    self.rope_theta = rope_theta
    self.q_proj = Linear(d_model, n_heads * self.head_dim, bias=False)
    self.k_proj = Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
    self.v_proj = Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
    self.o_proj = Linear(n_heads * self.head_dim, d_model, bias=False)
    self.q_norm = RMSNorm(self.head_dim, norm_eps) if qk_norm else None
    self.k_norm = RMSNorm(self.head_dim, norm_eps) if qk_norm else None
    self.record, self.recorded = False, None  # record=True keeps the last q and k, for drawing attention maps

  def forward(self, x, cache=None, pad=None):
    B, T, _ = x.shape
    D = self.head_dim
    if cache is not None and cache.rope is not None:  # the compiled decode step: positions arrive as arrays
      cos, sin = cache.rope
    else:
      start = 0 if cache is None else cache.n
      if pad is not None:
        start = start - np.asarray(pad)  # each row's first real token sits at position 0, after its left padding
      cos, sin = rope_tables(start, T, D, self.rope_theta)
    q = self.q_proj(x).reshape(B, T, self.n_heads, D)
    k = self.k_proj(x).reshape(B, T, self.n_kv_heads, D)
    if self.q_norm is not None:
      q, k = self.q_norm(q), self.k_norm(k)
    q, k = q.transpose(1, 2), k.transpose(1, 2)                                # (B, H, T, D), (B, Hkv, T, D)
    v = self.v_proj(x).reshape(B, T, self.n_kv_heads, D).transpose(1, 2)
    if self.rope:
      q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    if self.record:
      self.recorded = q.numpy(), k.numpy()
    if cache is not None:
      k, v = cache.update(k, v)
      if cache.rope is None:  # outside the compiled step, attend over the written positions only: the masked (padded) path scores every key
        k, v = k[:, :, :cache.n], v[:, :, :cache.n]
    out = scaled_dot_product_attention(q, k, v, pad, n=None if cache is None else cache.n)  # (B, H, T, D)
    return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))
