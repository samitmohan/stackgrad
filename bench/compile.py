import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import stackgrad.tensor as T
from stackgrad import no_grad
from stackgrad.attention import KVCache
from stackgrad.backend import metal
from stackgrad.qwen3 import load_qwen3

# Qwen3-0.6B in bfloat16 on the Apple GPU: decode with capture on and off (one chat, the path capture covers), and prefill and
# padded batched decode with fusion on and off (the paths capture skips). Modes alternate so drift hits both; medians of 3.
REPS, STEPS = 3, 32
CAPTURE, FUSABLE = metal.capture, set(T._FUSABLE)
model = load_qwen3("data/Qwen3-0.6B").to("metal", "bfloat16")
rng = np.random.default_rng(0)
one, prompt, rows = rng.integers(0, 151000, (1, 64)), rng.integers(0, 151000, (1, 512)), rng.integers(0, 151000, (8, 64))
pad = [0, 3, 7, 10, 15, 20, 30, 40]  # left padding, so batched decode carries pad and skips capture


def run(x, caches, pad=None):
  out = model.hidden(x, caches, pad).data
  mx.eval(out, *[a for c in caches for a in (c.k.data, c.v.data)])
  return out


def decode(x, pad=None):
  """ms per decode step after a prefill of x, and the last step's output."""
  with no_grad():
    caches = [KVCache() for _ in model.blocks]
    run(x, caches, pad)
    t0 = time.perf_counter()
    for _ in range(STEPS):
      out = run(x[:, -1:], caches, pad)
  return (time.perf_counter() - t0) / STEPS * 1e3, out


def prefill():
  with no_grad():
    t0 = time.perf_counter()
    out = run(prompt, [KVCache() for _ in model.blocks])
  return (time.perf_counter() - t0) * 1e3, out


def toggle(capture=True, fusion=True):
  metal.capture = CAPTURE if capture else None
  T._FUSABLE.clear()
  T._FUSABLE.update(FUSABLE if fusion else set())


def gap(a, b):
  return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


cases = {"decode_one_chat": (lambda: decode(one), "capture"), "prefill_512": (prefill, "fusion"),
         "decode_8_padded": (lambda: decode(rows, pad), "fusion")}
result = {"model": "Qwen3-0.6B", "dtype": "bfloat16", "steps": STEPS, "reps": REPS}
for name, (fn, switch) in cases.items():
  times, outs = {"off": [], "on": []}, {}
  for mode in ["off", "on"] * (REPS + 1):  # the first pair warms up and compiles
    toggle(**{switch: mode == "on"})
    ms, outs[mode] = fn()
    times[mode].append(ms)
  toggle()
  off, on = (statistics.median(times[m][1:]) for m in ("off", "on"))
  result[name] = {"switch": switch, "off_ms": [round(t, 2) for t in times["off"][1:]], "on_ms": [round(t, 2) for t in times["on"][1:]],
                  "median_off_ms": round(off, 2), "median_on_ms": round(on, 2), "speedup": round(off / on, 2),
                  "max_output_gap": gap(outs["off"], outs["on"])}
  print(f"{name:16s} {switch} off {off:7.2f} ms, on {on:7.2f} ms ({off / on:.2f}x), outputs differ by {result[name]['max_output_gap']:.3g}")
Path("bench/results/compile_metal.json").write_text(json.dumps(result, indent=1))
