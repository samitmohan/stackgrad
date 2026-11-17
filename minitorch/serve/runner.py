"""Model runners: the scheduler's only view of the model.

    prefill(seq_id, tokens) -> logits (V,)     run a whole prefix, keep its KV cache
    decode(seq_ids, tokens) -> logits (B, V)   one new token for each sequence, in one forward pass
    release(seq_id)                            drop that sequence's cache

Decode runs every sequence in one pass, so each weight is read once per step for the whole batch
instead of once per sequence. Their caches differ in length, so the runner left-pads them to the
longest and masks the padding (attention.py), as GRPO's batched sampling does. The padded caches
stay together while the batch keeps the same members; a join or a leave splits them back into
one cache per sequence and pads the new batch again.
"""
import numpy as np

from .. import backend
from ..attention import KVCache
from ..tensor import Tensor, no_grad


class MinitorchRunner:
    def __init__(self, model):
        self.model = model
        self.context_length = model.context_length
        self.caches = {}  # seq_id -> one KVCache per layer, for sequences outside the batch
        self.batch, self.batched, self.pad = (), None, None  # the batch's seq_ids, its padded caches, each row's padding

    def prefill(self, seq_id, tokens):
        self.caches[seq_id] = [KVCache() for _ in self.model.blocks]
        with no_grad():
            h = self.model.hidden(np.array([tokens]), self.caches[seq_id])
            return self.model.project(h[:, -1])[0].numpy()  # the last position only: the vocabulary-wide logits of the rest go unread

    def decode(self, seq_ids, tokens):
        # shortcut: a membership change copies every live cache twice (split, then pad); a paged attention kernel reading K/V through block tables avoids the copies
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
        """Move the sequences' caches into one batch, each left-padded to the longest."""
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
        """Give each sequence in the batch its own cache back, without its padding."""
        for b, s in enumerate(self.batch):
            p = int(self.pad[b])
            self.caches[s] = []
            for cache in self.batched:
                own = KVCache()
                own.k = Tensor(cache.k.data[b:b + 1, :, p:], device=cache.k.device)
                own.v = Tensor(cache.v.data[b:b + 1, :, p:], device=cache.v.device)
                self.caches[s].append(own)
        self.batch, self.batched, self.pad = (), None, None
