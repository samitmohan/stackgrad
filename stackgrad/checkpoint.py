from . import backend
from . import tensor as T
from .mlops import Function


class _Checkpoint(Function):
  def forward(self, *arrays, fn):
    self.fn, self.arrays, self.cast = fn, arrays, T._autocast
    device = self.parents[0].device
    with T.no_grad():
      out = fn(*[T.Tensor(a, device=device) for a in arrays]).data
    # realize now, or MLX runs it later inside one big eval that holds every activation
    self.xp.realize(out)
    return out

  def backward(self, grad):
    device = self.parents[0].device
    leaves = [T.Tensor(a, requires_grad=needs, device=device) for a, needs in zip(self.arrays, self.needs_grad)]
    with T.autocast(self.cast):  # forward's precision: backward usually runs outside autocast
      out = self.fn(*leaves)
    # backward of sum(out * grad) pushes grad through fn; weights inside fn collect .grad here
    (out * T.Tensor(grad, device=device)).sum().backward()
    # realize now, so this block's activations free before the next block recomputes
    weights = self.fn.parameters() if hasattr(self.fn, "parameters") else []
    self.xp.realize(*[t.grad for t in leaves + list(weights) if t.grad is not None])
    return tuple(leaf.grad for leaf in leaves)


def checkpoint(fn, *inputs):
  """fn(*inputs), recomputed during backward instead of storing its activations. fn must be deterministic."""
  if not T.grad_enabled():
    return fn(*inputs)
  device = inputs[0].device
  ctx = _Checkpoint(backend.get(device), *inputs)
  out = T.Tensor(ctx.forward(*[x.data for x in inputs], fn=fn), device=device)
  # always recorded: weights inside fn may need grads (LoRA adapters in frozen layers)
  out.requires_grad, out._ctx = True, ctx
  return out
