import numpy as np

from . import mlops
from .layers import Linear, RMSNorm
from .module import Module
from .tensor import _apply, cat


def apply_rope(x, start, theta):
  """Rotates each (x[i], x[i + D/2]) pair by its position's angle, in Llama/Qwen's half-split layout."""
  half = x.shape[-1] // 2
  freqs = theta ** (-np.arange(half) / half)
  positions = np.reshape(start, (-1, 1, 1)) + np.arange(x.shape[2])[:, None]  # (B or 1, T, 1)
  angles = (positions * freqs)[:, None]                                       # (B or 1, 1, T, D/2)
  cos, sin = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
  x1, x2 = x[..., :half], x[..., half:]
  return cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def scaled_dot_product_attention(q, k, v, pad=None, causal=True):
  """q (B, H, T, D) over k, v (B, Hkv, S, D), queries at the last T positions; pad (B,) masks left padding."""
  key_bias = None
  if pad is not None:
    # -1e9, not -inf: a padding query sees only padding, and an all -inf softmax is NaN that leaks via 0 * NaN
    key_bias = np.where(np.arange(k.shape[2]) < np.asarray(pad)[:, None], -1e9, 0.0).astype(np.float32)
  return _apply(mlops.Attention, q, k, v, key_bias=key_bias, causal=causal)


class KVCache:
  """Keys and values of earlier positions, so a decode step doesn't recompute the prefix."""
  # shortcut: cat copies the whole cache every step (O(T^2) bytes per sequence); M4 swaps in a paged cache
  def __init__(self):
    self.k = self.v = None

  def __len__(self):
    return 0 if self.k is None else self.k.shape[2]

  def truncate(self, n):
    """Drops every position from n on, for speculative decoding's rejected drafts."""
    self.k, self.v = self.k[:, :, :n], self.v[:, :, :n]

  def update(self, k, v):
    self.k = k if self.k is None else cat([self.k, k], axis=2)
    self.v = v if self.v is None else cat([self.v, v], axis=2)
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
    start = 0 if cache is None else len(cache)
    if pad is not None:
      start = start - np.asarray(pad)  # each row's first real token sits at position 0, after its left padding
    q = self.q_proj(x).reshape(B, T, self.n_heads, D)
    k = self.k_proj(x).reshape(B, T, self.n_kv_heads, D)
    if self.q_norm is not None:
      q, k = self.q_norm(q), self.k_norm(k)
    q, k = q.transpose(1, 2), k.transpose(1, 2)                                # (B, H, T, D), (B, Hkv, T, D)
    v = self.v_proj(x).reshape(B, T, self.n_kv_heads, D).transpose(1, 2)
    if self.rope:
      q, k = apply_rope(q, start, self.rope_theta), apply_rope(k, start, self.rope_theta)
    if cache is not None:
      k, v = cache.update(k, v)
    if self.record:
      self.recorded = q.numpy(), k.numpy()

    out = scaled_dot_product_attention(q, k, v, pad)                            # (B, H, T, D)
    return self.o_proj(out.transpose(1, 2).reshape(B, T, self.n_heads * D))
