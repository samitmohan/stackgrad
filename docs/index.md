# stackgrad documentation

stackgrad is a deep learning framework written from scratch, with everything built on it up to serving Qwen3-0.6B to a
retrieval app and a tool-using agent. This page is for users of the library. Install from source:

```bash
git clone https://github.com/samitmohan/stackgrad.git
cd stackgrad
uv sync --extra metal    # Apple silicon; NVIDIA: uv sync --extra cuda --no-dev
```

Then try the [quickstart](quickstart.md). The write-up, [Modern AI: Autograd to Agents](https://samitmohan.in/blog/minitorch),
explains each piece from first principles and measures it.

## stackgrad Usage

The main class you will use is [Tensor](tensor/index.md). It works like PyTorch's. All operations are lazy: they do
nothing until you read the result.

* stackgrad has a [neural network library](nn.md) with layers, losses, optimizers, and load/save.
* stackgrad runs on three [backends](runtime.md): NumPy, MLX on Apple's GPU, and CuPy on NVIDIA's, with attention
  kernels written in Metal and Triton.
* To see what training looks like, read `train_gpt.py`.

## Differences from PyTorch

### stackgrad is lazy

`a + b` records an op and returns a tensor that knows its shape and dtype but holds no numbers. Reading `.data`,
`.numpy()`, `.item()`, calling `.backward()`, or calling [`realize`](tensor/properties.md#stackgrad.tensor.Tensor.realize)
runs every recorded op the result depends on, once each.

### every gradient lives in one file

`stackgrad/mlops.py` defines 20 primitives (Add, Mul, Matmul, Sum, Reshape, Where, and the rest), each with a forward
and a backward. Every other op composes them: `relu` is a `Less` and a `Where`, so it never defines a gradient.

### devices are strings

A tensor lives on `"cpu"`, `"metal"`, or `"cuda"`. Pass `device=` to `Tensor`, or move a model with `model.to("metal")`.

### Tensor wraps arrays

There are no factory methods like `torch.zeros`. Build the array in NumPy and wrap it: `Tensor(np.zeros((2, 3)))`.
Float arrays keep their dtype; ints, bools, and Python numbers become float32. `.grad` holds a backend array, not a
Tensor.
