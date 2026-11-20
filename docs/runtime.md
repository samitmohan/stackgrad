# Runtime

stackgrad supports three backends. A tensor's `device` names one.

| Device | Backend | Kernels | Install |
|--------|---------|---------|---------|
| [cpu](https://github.com/samitmohan/minitorch/tree/master/stackgrad/backend/cpu.py) | NumPy, the reference every other backend is checked against | none | `uv sync` |
| [metal](https://github.com/samitmohan/minitorch/tree/master/stackgrad/backend/metal.py) | MLX on the Apple GPU | attention forward and backward, quantized matmul, in Metal through `mx.fast.metal_kernel` | `uv sync --extra metal` |
| [cuda](https://github.com/samitmohan/minitorch/tree/master/stackgrad/backend/cuda.py) | CuPy on an NVIDIA GPU | FlashAttention-2 forward and backward in Triton | `uv sync --extra cuda --no-dev` |

A backend is a module that exports 27 names: 25 array functions (`exp`, `log`, `where`, `broadcast_to`,
`scatter_add`, `im2col`, and the rest) plus `float32` and `inf`. `stackgrad/backend/cpu.py` lists them. `mlops.py` calls
them through `xp`, so every gradient is written once and runs on any backend. `backend.get(device)` imports a backend
on first use, since MLX only installs on macOS and CuPy only with CUDA.

`tests/backends.py --device metal` (or `cuda`) runs every op and a training step on the backend and compares each
against NumPy.

## MLX is lazy too

MLX records ops and runs them when a value is read. The optimizer realizes the weights after every step, or each
step's graph would hang off the last one. MLX arrays are immutable, so no gradient in `mlops.py` updates an array in
place.

## The attention contract

Every attention kernel takes `q (B, H, T, D)` and `k, v (B, Hkv, S, D)` and returns `out (B, H, T, D)` and
`lse (B, H, T)`. Query row `i` sits at absolute position `S - T + i` and sees keys `0 .. S - T + i`, so one kernel
serves training (`T == S`) and decoding against a KV cache (`T < S`). Query head `h` reads KV head `h // (H / Hkv)`,
which is grouped-query attention without copying K and V. The scale is `1/sqrt(D)`. `lse` is each row's log-sum-exp
of scores, all the backward pass needs to rebuild the softmax. No kernel writes the `(T, S)` score matrix to memory.

## Environment Variables

| Variable | Effect |
|----------|--------|
| `CUPY_TF32=1` | CuPy's matmuls and the Triton attention kernel use TF32 instead of full float32 |
