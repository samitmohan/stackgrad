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
    lengths = np.array([r[0].n for r in rows])
    self.pad = lengths.max() - lengths
    device = rows[0][0].k.device
    xp = backend.get(device)

    def padded(layer, name):
      parts = [xp.pad(getattr(r[layer], name).data[:, :, :n], ((0, 0), (0, 0), (p, 0), (0, 0)))
               for r, n, p in zip(rows, lengths.tolist(), self.pad.tolist())]
      return Tensor(xp.concatenate(parts), device=device)

    self.batched = [KVCache(padded(i, "k"), padded(i, "v"), int(lengths.max())) for i in range(len(self.model.blocks))]
    self.batch = tuple(seq_ids)

  def _split(self):
    for b, s in enumerate(self.batch):
      p = int(self.pad[b])
      self.caches[s] = [KVCache(*(Tensor(t.data[b:b + 1, :, p:c.n], device=t.device) for t in (c.k, c.v)), c.n - p) for c in self.batched]
    self.batch, self.batched, self.pad = (), None, None
