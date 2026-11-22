import json
import sys
from pathlib import Path

import numpy as np

from stackgrad import Tensor, no_grad
from stackgrad.attention import KVCache
from stackgrad.qwen3 import load_qwen3

# Peak GPU memory of a Qwen3-0.6B prefill (16 rows x 96 tokens): realize as it is, which frees each buffer once its last
# reader has run, against holding every node of the graph until realize returns, which is what realize used to do.
DEVICE = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "metal"
model = load_qwen3("data/Qwen3-0.6B").to(DEVICE)
x = np.random.default_rng(0).integers(0, 151000, (16, 96))
realize = Tensor.realize
if DEVICE == "metal":
  import mlx.core as mx
  start, peak_bytes = mx.reset_peak_memory, mx.get_peak_memory
else:
  import cupy as cp
  pool = cp.get_default_memory_pool()
  start, peak_bytes = pool.free_all_blocks, pool.total_bytes  # the pool grows to the peak it needed and keeps it


def graph(t):
  nodes, seen, stack = [], set(), [t]
  while stack:
    node = stack.pop()
    if node._buf is None and id(node) not in seen:
      seen.add(id(node))
      nodes.append(node)
      stack.extend(i for i in node._lazy[2] if isinstance(i, Tensor))
  return nodes


def hold_all(self):
  held = graph(self)  # the old realize kept these referenced, so none of their buffers could go before it returned
  return realize(self) if held else self


def peak(fn):
  Tensor.realize = fn
  start()
  base = peak_bytes()
  with no_grad():
    out = model.hidden(x, [KVCache() for _ in model.blocks]).realize().numpy()
  Tensor.realize = realize
  return (peak_bytes() - base) / 1e9, out


new, a = peak(realize)
old, b = peak(hold_all)
gap = float(np.abs(a - b).max())
assert gap < 1e-3, gap  # same graph, same numbers
result = {"model": "Qwen3-0.6B", "device": DEVICE, "dtype": "float32", "rows": 16, "tokens": 96, "max_output_gap": gap,
          "peak_gb_above_start": {"free_as_you_go": round(new, 2), "hold_until_return": round(old, 2)}}
print(result)
Path(f"bench/results/realize_memory_{DEVICE}.json").write_text(json.dumps(result, indent=1))
