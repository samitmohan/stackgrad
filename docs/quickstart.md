# Quick Start Guide

This guide assumes you know what a neural network is, and nothing about stackgrad. By the end you will have trained a
small classifier, saved it, and moved it to a GPU.

## Tensors

A tensor wraps an array. Build one from a Python list or a NumPy array:

```python
import numpy as np
from stackgrad import Tensor

t1 = Tensor([1, 2, 3, 4, 5])
t2 = Tensor(np.arange(6.0).reshape(2, 3))
print(t1.dtype, t2.shape)  # float32 (2, 3)
```

Ints become float32. Float arrays keep their dtype, so `t2` is float64.

## Operations

Arithmetic operators, matrix multiply, reductions, and shape ops work as in PyTorch:

```python
t3 = (t2 * 2 + 1).relu()
t4 = t2 @ t2.T
print(t3.numpy())
print(t4.sum(axis=1).numpy())
```

Nothing runs until you read a result. `t3` held no numbers until `.numpy()` asked for them.

## Autograd

Mark the tensors you want gradients for, compute a scalar, and call `backward`:

```python
w = Tensor(np.random.randn(3, 1), requires_grad=True)
loss = ((t2 @ w) ** 2).mean()
loss.backward()
print(w.grad.shape)  # (3, 1)
```

`w.grad` is a NumPy array on the CPU backend.

## Models

Subclass `Module`, assign layers as attributes, and write `forward`. `parameters()` finds every trainable tensor:

```python
from stackgrad import Module, Linear

class TinyNet(Module):
  def __init__(self):
    super().__init__()
    self.l1, self.l2 = Linear(2, 16), Linear(16, 2)

  def forward(self, x):
    return self.l2(self.l1(x).relu())

net = TinyNet()
print(sum(p.data.size for p in net.parameters()))  # 82
```

## Training

The task: given a point in the plane, say whether its two coordinates have the same sign.

```python
from stackgrad import AdamW, cross_entropy_loss

X = np.random.randn(512, 2).astype(np.float32)
Y = (X[:, 0] * X[:, 1] > 0).astype(np.int64)
opt = AdamW(net.parameters(), lr=1e-2)

for step in range(300):
  idx = np.random.randint(0, len(X), 64)
  loss = cross_entropy_loss(net(Tensor(X[idx])), Y[idx])
  opt.zero_grad()
  loss.backward()
  opt.step()
  if step % 100 == 0:
    print(step, loss.item())
```

## Evaluation

`no_grad` stops the graph from recording what backward would need:

```python
from stackgrad import no_grad

with no_grad():
  acc = (net(Tensor(X)).numpy().argmax(axis=1) == Y).mean()
print(f"accuracy {acc:.2f}")
```

## Saving and loading

```python
net.save("tinynet.npz")
net.load("tinynet.npz")
```

## Devices

`to` moves every parameter in place, so an optimizer built before the move still holds the same tensors. Inputs go on
the same device:

```python
net.to("metal")  # needs uv sync --extra metal; "cuda" needs --extra cuda
with no_grad():
  print(net(Tensor(X[:4], device="metal")).numpy())
```

For a real model, read `train_gpt.py`: it trains a 22.7M-parameter transformer on TinyStories with the same pieces.
