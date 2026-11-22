import importlib.util
import inspect
import sys

import numpy as np

from stackgrad import TransformerLM, cross_entropy_loss, mlops, no_grad
from stackgrad import tensor as T
from stackgrad.tensor import Tensor, where

RAN = []
for name, fn in inspect.getmembers(mlops, inspect.isclass):
  if issubclass(fn, mlops.Function) and fn is not mlops.Function and "forward" in fn.__dict__:
    def counted(self, *args, _forward=fn.forward, _name=name, **kwargs):
      RAN.append(_name)
      return _forward(self, *args, **kwargs)
    fn.forward = counted

failures = 0


def check(ok, what):
  global failures
  failures += not ok
  print(f"[{'ok' if ok else 'FAIL'}]   {what}")


rng = np.random.default_rng(0)
A, B = rng.standard_normal((64, 32)), rng.standard_normal((32, 16))
a, b = Tensor(A), Tensor(B)

c = a @ b
check(RAN == [] and c.shape == (64, 16) and c.dtype == np.float64 and c._buf is None,
      f"a @ b runs nothing and still knows its shape {c.shape} and dtype {c.dtype}")


def layer(x, w):
  return (x @ w + 1.0).relu().sum(axis=0)


out = layer(a, b)
unused = (a @ a.T).exp()
check(RAN == [], "a function that builds and returns a result runs nothing")

value = out.numpy()
ran = list(RAN)
check(sorted(ran) == sorted(["Matmul", "Add", "Less", "Where", "Sum"]) and len(ran) == 5,
      f"reading the result runs each op it needs once: {ran}")
check(np.array_equal(value, np.where(A @ B + 1.0 > 0, A @ B + 1.0, 0.0).sum(axis=0)), "the result equals NumPy's, bit for bit")
out.numpy()
check(RAN == ran, "a second read runs nothing")
check("Exp" not in RAN and unused._buf is None, "an op nobody reads never runs")

w = Tensor(B.copy())
y = a @ w
w.data = np.zeros_like(B)  # as an optimizer step replaces a weight's array
check(np.array_equal(y.numpy(), A @ B), "replacing an input's .data after the op was written leaves its result as written")

x = Tensor(rng.standard_normal(10), requires_grad=True)
mask = x > 0
check(mask._buf is None and where(mask, x, 0.0)._buf is None, "comparisons and where() are recorded, not run")

RAN.clear()
loss = (where(x > 0, x, 0.0) * x).sum()
check(RAN == [], "a loss is recorded, not run")
loss.backward()
check(np.allclose(x.grad, np.where(x.data > 0, 2 * x.data, 0.0)), "backward on a lazy loss runs it and gives d/dx of relu(x) * x")

model = TransformerLM(97, 16, 32, 2, 4, 64)
RAN.clear()
logits = model(rng.integers(0, 97, (2, 16)))
check(RAN == [] and logits.shape == (2, 16, 97), f"a TransformerLM forward runs nothing until read: {RAN[:5]}")


def graphs(x, y):
  h = (x * 3.0).exp()
  return {"chain": ((x * 2.0 + 1.0).exp().log() * x).sigmoid(), "branch": (h + 1.0) * (h - 1.0),
          "broadcast": (x * y.expand(*x.shape)).exp() + y, "cast": (x * 2.0).cast("float16").exp().cast("float32") + 1.0,
          "where": where(x > 0, x * 2.0, x.exp())}


def evaluate(device, fuse):
  T._FUSABLE = FUSABLE if fuse else set()
  with no_grad():
    out = {k: v.numpy() for k, v in graphs(Tensor(X, device=device), Tensor(Y, device=device)).items()}
  T._FUSABLE = FUSABLE
  return out


FUSABLE, X, Y = T._FUSABLE, rng.standard_normal((256, 256)).astype(np.float32), rng.standard_normal(256).astype(np.float32)  # 2^16: fusion's floor
for device in ["cpu"] + (["metal"] if importlib.util.find_spec("mlx") else []):
  before = T._fused.cache_info()
  fused, plain = evaluate(device, True), evaluate(device, False)
  misses = T._fused.cache_info().misses - before.misses
  same = {k: np.array_equal(fused[k], plain[k]) if device == "cpu" else np.allclose(fused[k], plain[k], rtol=1e-5, atol=1e-6) for k in fused}
  check(all(same.values()) and misses == len(fused), f"{device}: fused equals unfused ({'bitwise' if device == 'cpu' else 'allclose'}), "
        f"{misses} runs compiled: {same}")
  before = T._fused.cache_info()
  again = evaluate(device, True)
  after = T._fused.cache_info()
  check(after.misses == before.misses and after.hits - before.hits == len(fused) and all(np.array_equal(again[k], fused[k]) for k in fused),
        f"{device}: the same graphs built again compile nothing ({after.misses - before.misses} misses, {after.hits - before.hits} hits)")

with no_grad():
  h = Tensor(X).exp()
  out = h * 2.0 + 1.0
  out.realize()
  internal = h._buf is None
  check(internal and np.array_equal(h.numpy(), np.exp(X)) and np.array_equal(out.numpy(), np.exp(X) * np.float32(2) + np.float32(1)),
        "an intermediate a fused run kept inside recomputes, bit for bit, when read afterwards")


def train_step(fuse):
  T._FUSABLE = FUSABLE if fuse else set()
  np.random.seed(0)
  model = TransformerLM(97, 16, 32, 2, 4, 64)
  scale = Tensor(np.random.default_rng(1).standard_normal(1 << 16).astype(np.float32))
  before = T._fused.cache_info()
  # (scale > 0) * 0.5 + 1.0 needs no grad, so it fuses inside a graph that trains
  loss = cross_entropy_loss(model(tokens) * ((scale > 0) * 0.5 + 1.0).mean(), tokens)
  loss.backward()
  calls = sum(T._fused.cache_info()[:2]) - sum(before[:2])
  T._FUSABLE = FUSABLE
  return [loss.numpy()] + [p.grad for p in model.parameters()], calls


tokens = rng.integers(0, 97, (2, 16))
on, fused_calls = train_step(True)
off, plain_calls = train_step(False)
check(fused_calls > 0 and plain_calls == 0 and len(on) == len(off) and all(np.array_equal(a, b) for a, b in zip(on, off)),
      f"a training step's loss and {len(on) - 1} gradients are bitwise equal with fusion on ({fused_calls} fused runs) and off")

print(f"\n{'all lazy checks passed' if failures == 0 else f'{failures} lazy checks failed'}")
sys.exit(1 if failures else 0)
