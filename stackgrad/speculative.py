import numpy as np

from .attention import KVCache
from .tensor import no_grad


def _probs(logits, temperature):
  if temperature == 0:
    p = np.zeros_like(logits)
    p[np.arange(len(logits)), logits.argmax(axis=-1)] = 1.0
    return p
  z = logits / temperature
  p = np.exp(z - z.max(axis=-1, keepdims=True))
  return p / p.sum(axis=-1, keepdims=True)


def _draw(p):
  return min(int(np.searchsorted(np.cumsum(p), np.random.rand() * p.sum(), side="right")), len(p) - 1)


def _scores(model, caches, tokens, last):
  h = model.hidden(np.array([tokens]), caches)
  return model.project(h[:, -last:])[0].numpy().astype(np.float64)


def speculative_generate(target, draft, ids, max_new, k=4, temperature=0.0, eos_id=None):
  """Up to max_new target tokens after ids, draft proposing k at a time; returns (tokens, proposed, accepted)."""
  assert len(ids) + max_new + k + 1 <= target.context_length, "prompt + new tokens exceed context_length"
  t_caches = [KVCache() for _ in target.blocks]
  d_caches = [KVCache() for _ in draft.blocks]
  out, proposed, accepted = [], 0, 0
  with no_grad():
    # each cache holds every token except the model's pending ones
    if len(ids) > 1:
      _scores(target, t_caches, ids[:-1], 1)
      _scores(draft, d_caches, ids[:-1], 1)
    pending_t, pending_d = [ids[-1]], [ids[-1]]
    while len(out) < max_new:
      drafts, qs = [], []
      q = _probs(_scores(draft, d_caches, pending_d, 1), temperature)[0]
      for i in range(k):
        drafts.append(_draw(q))
        qs.append(q)
        if i < k - 1:
          q = _probs(_scores(draft, d_caches, [drafts[-1]], 1), temperature)[0]
      # one target pass over its pending token and every draft: k + 1 distributions
      p = _probs(_scores(target, t_caches, pending_t + drafts, k + 1), temperature)
      # keep draft x with prob min(1, p/q), else resample max(0, p - q): exact target distribution
      n = 0
      while n < k and np.random.rand() < min(1.0, p[n, drafts[n]] / qs[n][drafts[n]]):
        n += 1
      last = _draw(np.maximum(p[n] - qs[n], 0.0)) if n < k else _draw(p[k])
      proposed, accepted = proposed + k, accepted + n

      # roll both caches back to the kept tokens (the target saw every draft, the draft all but the last); later writes overwrite the rest
      for c in t_caches:
        c.n -= k - n
      for c in d_caches:
        c.n -= k - 1 - min(n, k - 1)
      pending_t = [last]
      pending_d = ([drafts[-1]] if n == k else []) + [last]

      new = drafts[:n] + [last]
      if eos_id is not None and eos_id in new:
        out += new[:new.index(eos_id)]
        break
      out += new
  return out[:max_new], proposed, accepted
