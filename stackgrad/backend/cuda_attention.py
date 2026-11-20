import math
import os

import cupy as cp
import triton
import triton.language as tl

# match CuPy's matmuls: tf32 only under CUPY_TF32=1 (training), ieee float32 otherwise (tests)
_PRECISION = "tf32" if os.environ.get("CUPY_TF32") == "1" else "ieee"


class _Arg:
  """The data_ptr and dtype Triton reads from a tensor, so CuPy arrays need no torch."""
  def __init__(self, a):
    self.a, self.dtype = a, a.dtype

  def data_ptr(self):
    return self.a.data.ptr


@triton.jit
def _forward(Q, K, V, O, LSE, T, S, H, HKV, scale, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
  qb, bh = tl.program_id(0), tl.program_id(1)
  kv = (bh // H * HKV + bh % H // (H // HKV)) * S * D
  rows = qb * BLOCK_M + tl.arange(0, BLOCK_M)
  dims = tl.arange(0, D)
  q = tl.load(Q + (bh * T + rows[:, None]) * D + dims[None, :], mask=rows[:, None] < T, other=0.0)
  off = S - T
  m = tl.full([BLOCK_M], float("-inf"), tl.float32)
  l = tl.zeros([BLOCK_M], tl.float32)
  acc = tl.zeros([BLOCK_M, D], tl.float32)
  for start in range(0, tl.minimum(off + qb * BLOCK_M + BLOCK_M, S), BLOCK_N):  # later keys are hidden from every row here
    cols = start + tl.arange(0, BLOCK_N)
    k = tl.load(K + kv + cols[:, None] * D + dims[None, :], mask=cols[:, None] < S, other=0.0)
    v = tl.load(V + kv + cols[:, None] * D + dims[None, :], mask=cols[:, None] < S, other=0.0)
    s = tl.dot(q, tl.trans(k), input_precision=PREC) * scale
    s = tl.where((cols[None, :] <= off + rows[:, None]) & (cols[None, :] < S), s, float("-inf"))
    m_new = tl.maximum(m, tl.max(s, 1))
    p = tl.exp(s - m_new[:, None])
    corr = tl.exp(m - m_new)
    l = l * corr + tl.sum(p, 1)
    acc = acc * corr[:, None] + tl.dot(p, v, input_precision=PREC)
    m = m_new
  tl.store(O + (bh * T + rows[:, None]) * D + dims[None, :], acc / l[:, None], mask=rows[:, None] < T)
  tl.store(LSE + bh * T + rows, m + tl.log(l), mask=rows < T)


@triton.jit
def _backward_kv(Q, K, V, DO, LSE, DELTA, DK, DV, T, S, H, HKV, scale,
                 D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
  kb, bh = tl.program_id(0), tl.program_id(1)
  kv = (bh // H * HKV + bh % H // (H // HKV)) * S * D
  cols = kb * BLOCK_N + tl.arange(0, BLOCK_N)
  dims = tl.arange(0, D)
  k = tl.load(K + kv + cols[:, None] * D + dims[None, :], mask=cols[:, None] < S, other=0.0)
  v = tl.load(V + kv + cols[:, None] * D + dims[None, :], mask=cols[:, None] < S, other=0.0)
  off = S - T
  dk = tl.zeros([BLOCK_N, D], tl.float32)
  dv = tl.zeros([BLOCK_N, D], tl.float32)
  first = tl.maximum(kb * BLOCK_N - off, 0) // BLOCK_M * BLOCK_M  # earlier query rows see none of these keys
  for start in range(first, T, BLOCK_M):
    rows = start + tl.arange(0, BLOCK_M)
    q = tl.load(Q + (bh * T + rows[:, None]) * D + dims[None, :], mask=rows[:, None] < T, other=0.0)
    do = tl.load(DO + (bh * T + rows[:, None]) * D + dims[None, :], mask=rows[:, None] < T, other=0.0)
    lse = tl.load(LSE + bh * T + rows, mask=rows < T, other=0.0)
    delta = tl.load(DELTA + bh * T + rows, mask=rows < T, other=0.0)
    visible = (cols[None, :] <= off + rows[:, None]) & (rows[:, None] < T) & (cols[None, :] < S)
    p = tl.where(visible, tl.exp(tl.dot(q, tl.trans(k), input_precision=PREC) * scale - lse[:, None]), 0.0)
    dv += tl.dot(tl.trans(p), do, input_precision=PREC)
    ds = p * (tl.dot(do, tl.trans(v), input_precision=PREC) - delta[:, None])
    dk += tl.dot(tl.trans(ds), q, input_precision=PREC) * scale
  # per query head; the caller sums each group into its kv head
  tl.store(DK + (bh * S + cols[:, None]) * D + dims[None, :], dk, mask=cols[:, None] < S)
  tl.store(DV + (bh * S + cols[:, None]) * D + dims[None, :], dv, mask=cols[:, None] < S)


@triton.jit
def _backward_q(Q, K, V, DO, LSE, DELTA, DQ, T, S, H, HKV, scale, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
  qb, bh = tl.program_id(0), tl.program_id(1)
  kv = (bh // H * HKV + bh % H // (H // HKV)) * S * D
  rows = qb * BLOCK_M + tl.arange(0, BLOCK_M)
  dims = tl.arange(0, D)
  q = tl.load(Q + (bh * T + rows[:, None]) * D + dims[None, :], mask=rows[:, None] < T, other=0.0)
  do = tl.load(DO + (bh * T + rows[:, None]) * D + dims[None, :], mask=rows[:, None] < T, other=0.0)
  lse = tl.load(LSE + bh * T + rows, mask=rows < T, other=0.0)
  delta = tl.load(DELTA + bh * T + rows, mask=rows < T, other=0.0)
  off = S - T
  dq = tl.zeros([BLOCK_M, D], tl.float32)
  for start in range(0, tl.minimum(off + qb * BLOCK_M + BLOCK_M, S), BLOCK_N):
    cols = start + tl.arange(0, BLOCK_N)
    k = tl.load(K + kv + cols[:, None] * D + dims[None, :], mask=cols[:, None] < S, other=0.0)
    v = tl.load(V + kv + cols[:, None] * D + dims[None, :], mask=cols[:, None] < S, other=0.0)
    visible = (cols[None, :] <= off + rows[:, None]) & (rows[:, None] < T) & (cols[None, :] < S)
    p = tl.where(visible, tl.exp(tl.dot(q, tl.trans(k), input_precision=PREC) * scale - lse[:, None]), 0.0)
    ds = p * (tl.dot(do, tl.trans(v), input_precision=PREC) - delta[:, None])
    dq += tl.dot(ds, k, input_precision=PREC) * scale
  tl.store(DQ + (bh * T + rows[:, None]) * D + dims[None, :], dq, mask=rows[:, None] < T)


def _blocks(D):
  # D=128 at 64x32 blocks needs 114,688 B of shared memory; the 3090 has 101,376
  return (64, 64) if D <= 64 else (32, 32)


def supported(q):
  D = q.shape[-1]
  return q.dtype == cp.float32 and 16 <= D <= 128 and D & (D - 1) == 0  # tl.arange needs a power of two


def attention(q, k, v):
  """Causal FlashAttention-2: q (B, H, T, D), k, v (B, Hkv, S, D) -> out (B, H, T, D), lse (B, H, T)."""
  B, H, T, D = q.shape
  S, Hkv = k.shape[2], k.shape[1]
  q, k, v = (cp.ascontiguousarray(x) for x in (q, k, v))
  out, lse = cp.empty_like(q), cp.empty((B, H, T), cp.float32)
  bm, bn = _blocks(D)
  _forward[(triton.cdiv(T, bm), B * H)](_Arg(q), _Arg(k), _Arg(v), _Arg(out), _Arg(lse), T, S, H, Hkv, 1 / math.sqrt(D),
                                        D=D, BLOCK_M=bm, BLOCK_N=bn, PREC=_PRECISION)
  return out, lse


def attention_backward(q, k, v, out, lse, dout):
  """(dq, dk, dv) of attention(q, k, v)[0] given dout, rebuilding the softmax from out and lse."""
  B, H, T, D = q.shape
  S, Hkv = k.shape[2], k.shape[1]
  q, k, v, out, dout = (cp.ascontiguousarray(x) for x in (q, k, v, out, dout))
  delta = (dout * out).sum(axis=-1)  # rowsum(dO * O), the softmax backward's correction term
  dq = cp.empty_like(q)
  dk, dv = cp.empty((B, H, S, D), cp.float32), cp.empty((B, H, S, D), cp.float32)
  bm, bn = _blocks(D)
  args = (T, S, H, Hkv, 1 / math.sqrt(D))
  _backward_kv[(triton.cdiv(S, bn), B * H)](*map(_Arg, (q, k, v, dout, lse, delta, dk, dv)), *args, D=D, BLOCK_M=bm, BLOCK_N=bn, PREC=_PRECISION,
                                            num_stages=1)
  # one pipeline stage: the backward holds more tiles, and three stages overflow shared memory
  _backward_q[(triton.cdiv(T, bm), B * H)](*map(_Arg, (q, k, v, dout, lse, delta, dq)), *args, D=D, BLOCK_M=bm, BLOCK_N=bn, PREC=_PRECISION,
                                           num_stages=1)
  rep = H // Hkv
  return dq, dk.reshape(B, Hkv, rep, S, D).sum(axis=2), dv.reshape(B, Hkv, rep, S, D).sum(axis=2)
