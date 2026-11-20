import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stackgrad import Tensor, no_grad
from stackgrad.backend.metal_quant import quant_matmul
from stackgrad.quant import QuantLinear, dequantize, quantize


def main():
  rng = np.random.default_rng(0)
  w = rng.standard_normal((96, 256)).astype(np.float32)
  errors = {}
  for bits in (4, 8):
    packed, s, b = quantize(w, bits, 64)
    mine = dequantize(packed, s, b, bits, 64, np)
    theirs = np.array(mx.dequantize(mx.array(packed), mx.array(s), mx.array(b), group_size=64, bits=bits))
    assert np.abs(mine - theirs).max() < 1e-5, f"{bits}-bit layout differs from MLX's"
    errors[bits] = float(np.abs(mine - w).max())
  assert errors[8] < errors[4] / 8, errors
  print(f"[ok]   packing matches mx.dequantize; largest rounding error int4 {errors[4]:.3f}, int8 {errors[8]:.4f}")

  worst = 0.0
  for bits in (4, 8):
    packed, s, b = quantize(w, bits, 64)
    for dtype in (mx.float32, mx.bfloat16):
      for m in (1, 3, 8):
        x = mx.array(rng.standard_normal((m, 256)).astype(np.float32)).astype(dtype)
        args = (mx.array(packed), mx.array(s).astype(dtype), mx.array(b).astype(dtype))
        got = quant_matmul(x, *args, bits, 64).astype(mx.float32)
        want = mx.quantized_matmul(x, *args, transpose=True, group_size=64, bits=bits).astype(mx.float32)
        rel = float(mx.abs(got - want).max() / mx.abs(want).max())
        worst = max(worst, rel)
        assert rel < (2e-2 if dtype == mx.bfloat16 else 1e-5), (bits, dtype, m, rel)
  print(f"[ok]   Metal kernel matches mx.quantized_matmul (largest relative gap {worst:.1e}, bfloat16 included)")

  weight = Tensor(w.T)  # stackgrad's (in, out)
  layers = {"cpu": QuantLinear(weight, 4, 64), "metal": QuantLinear(weight.to("metal"), 4, 64)}
  for rows in (1, 20):  # 1 takes the Metal kernel, 20 the dequantize-then-matmul path
    x = rng.standard_normal((rows, 256)).astype(np.float32)
    with no_grad():
      out = {d: layer(Tensor(x, device=d)).numpy() for d, layer in layers.items()}
    gap = float(np.abs(out["cpu"] - out["metal"]).max())
    assert gap < 1e-3, (rows, gap)
  print("[ok]   QuantLinear agrees on NumPy and Metal for decode and prefill shapes")


if __name__ == "__main__":
  main()
