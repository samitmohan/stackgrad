class BlockManager:
  """Paged KV-cache bookkeeping: which fixed-size blocks of cache memory each sequence owns."""
  def __init__(self, num_blocks, block_size=16):
    self.block_size = block_size
    self.num_blocks = num_blocks
    self.free = list(range(num_blocks - 1, -1, -1))  # pop() hands out block 0 first
    self.tables = {}   # seq_id -> [block ids]
    self.lengths = {}  # seq_id -> tokens stored

  def blocks_needed(self, seq_id, n_new):
    """How many more blocks `seq_id` needs to store `n_new` more tokens."""
    have = len(self.tables.get(seq_id, ()))
    need = -(-(self.lengths.get(seq_id, 0) + n_new) // self.block_size)  # ceil
    return max(0, need - have)

  def can_append(self, seq_id, n_new):
    return self.blocks_needed(seq_id, n_new) <= len(self.free)

  def append(self, seq_id, n_new):
    """Record `n_new` more tokens for `seq_id`, taking blocks as needed. Returns its block table."""
    n_blocks = self.blocks_needed(seq_id, n_new)
    if n_blocks > len(self.free):
      raise MemoryError(f"sequence {seq_id} needs {n_blocks} blocks, {len(self.free)} free")
    table = self.tables.setdefault(seq_id, [])
    table.extend(self.free.pop() for _ in range(n_blocks))
    self.lengths[seq_id] = self.lengths.get(seq_id, 0) + n_new
    return table

  def release(self, seq_id):
    """Return every block of `seq_id` to the pool."""
    self.free.extend(reversed(self.tables.pop(seq_id, [])))
    self.lengths.pop(seq_id, None)

  def used(self):
    return self.num_blocks - len(self.free)
