import numpy as np

from .. import backend
from ..attention import KVCache
from ..tensor import Tensor, no_grad


class StackgradRunner:
  """The scheduler's view of the model: prefill one sequence, decode the batch over left-padded KV caches."""
  def __init__(self, model):
    self.model = model
    self.context_length = model.context_length
    self.caches = {}  # seq_id -> one KVCache per layer, for sequences outside the batch
    self.batch, self.batched, self.pad = (), None, None  # the batch's seq_ids, its padded caches, each row's padding

  def prefill(self, seq_id, tokens):
    self.caches[seq_id] = [KVCache() for _ in self.model.blocks]
    with no_grad():
      h = self.model.hidden(np.array([tokens]), self.caches[seq_id])
      return self.model.project(h[:, -1])[0].numpy()  # last position only: the other logits go unread

  def decode(self, seq_ids, tokens):
    # shortcut: a membership change copies every live cache twice (split, pad); paged attention over block tables avoids it
    if tuple(seq_ids) != self.batch:
      self._split()
      self._join(seq_ids)
    with no_grad():
      logits = self.model(np.array(tokens)[:, None], self.batched, self.pad if self.pad.any() else None)
      return logits[:, -1].numpy()

  def release(self, seq_id):
    if seq_id in self.batch:
      self._split()
    self.caches.pop(seq_id, None)

  def _join(self, seq_ids):
    rows = [self.caches.pop(s) for s in seq_ids]
    lengths = np.array([len(r[0]) for r in rows])
    self.pad = lengths.max() - lengths
    device = rows[0][0].k.device
    xp = backend.get(device)

    def padded(layer, name):
      parts = [xp.pad(getattr(r[layer], name).data, ((0, 0), (0, 0), (int(p), 0), (0, 0))) for r, p in zip(rows, self.pad)]
      return Tensor(xp.concatenate(parts), device=device)

    self.batched = []
    for i in range(len(self.model.blocks)):
      cache = KVCache()
      cache.k, cache.v = padded(i, "k"), padded(i, "v")
      self.batched.append(cache)
    self.batch = tuple(seq_ids)

  def _split(self):
    for b, s in enumerate(self.batch):
      p = int(self.pad[b])
      self.caches[s] = []
      for cache in self.batched:
        own = KVCache()
        own.k = Tensor(cache.k.data[b:b + 1, :, p:], device=cache.k.device)
        own.v = Tensor(cache.v.data[b:b + 1, :, p:], device=cache.v.device)
        self.caches[s].append(own)
    self.batch, self.batched, self.pad = (), None, None
