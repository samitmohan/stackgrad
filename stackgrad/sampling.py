import numpy as np

from . import backend


def sample(logits, temperature=1.0, top_k=None, top_p=None):
  """One token id per row of logits (B, V); temperature=0 is greedy, top_k and top_p filter first."""
  if temperature == 0:
    return logits.argmax(axis=-1)
  logits = logits / temperature
  if top_k is not None:
    kth = np.sort(logits, axis=-1)[:, -top_k][:, None]
    logits = np.where(logits < kth, -np.inf, logits)
  probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
  probs /= probs.sum(axis=-1, keepdims=True)
  if top_p is not None:
    kept = np.zeros_like(probs)
    for row, out in zip(probs, kept):
      # tokens under (1 - top_p) / V sum to under 1 - top_p, so the nucleus lies above: sort only those
      cand = np.flatnonzero(row >= (1 - top_p) / len(row))
      cand = cand[np.argsort(-row[cand])]
      # drop a token once the likelier tokens before it already reach top_p
      keep = cand[np.cumsum(row[cand]) - row[cand] < top_p]
      out[keep] = row[keep]
    probs = kept / kept.sum(axis=-1, keepdims=True)
  return np.array([np.random.choice(len(p), p=p) for p in probs])


def sample_on_device(logits, temperature=1.0):
  """sample() without top-k or top-p, run where the Tensor of logits lives, so only B ids reach the host."""
  xp = backend.get(logits.device)
  z = logits.data.astype(xp.float32)
  if temperature == 0:
    return xp.to_numpy(z.argmax(axis=-1))
  z = z / temperature
  cdf = xp.cumsum(xp.exp(z - z.max(axis=-1, keepdims=True)), axis=-1)
  u = xp.asarray(np.random.rand(len(z), 1).astype(np.float32)) * cdf[:, -1:]
  # first crossing, not a count of cdf < u: a parallel float32 cumsum dips. u < cdf[-1], so one exists
  return xp.to_numpy((cdf >= u).argmax(axis=-1))
