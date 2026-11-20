import os
from pathlib import Path

import numpy as np

from . import backend
from .tensor import Tensor


class Module:
  """Base class for layers and models; Tensor and Module attributes are found by parameters() and state_dict()."""
  def __init__(self):
    self._training = True

  def forward(self, *args, **kwargs):
    raise NotImplementedError

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def _attrs(self):
    for name, val in self.__dict__.items():
      if isinstance(val, (list, tuple)):
        yield from ((f"{name}.{i}", m) for i, m in enumerate(val) if isinstance(m, Module))
      else:
        yield name, val

  def parameters(self):
    """Collect every trainable `Tensor` in this module and its children."""
    return list({id(p): p for _, p in self.named_parameters()}.values())  # tied weights must not step twice

  def named_parameters(self, prefix=""):
    """(dotted name, tensor) for every trainable tensor, named as in state_dict()."""
    for name, val in self._attrs():
      if isinstance(val, Tensor) and val.requires_grad:
        yield prefix + name, val
      elif isinstance(val, Module):
        yield from val.named_parameters(f"{prefix}{name}.")

  def train(self, mode=True):
    self._training = mode
    for _, val in self._attrs():
      if isinstance(val, Module):
        val.train(mode)
    return self

  def eval(self):
    return self.train(False)

  def to(self, device=None, dtype=None):
    """Move every tensor to `device` and/or cast it to `dtype` in place, so optimizers keep their references."""
    moved = []
    for val in self.__dict__.values():
      items = val if isinstance(val, (list, tuple)) else [val]
      for item in items:
        if isinstance(item, Tensor):
          copy = item.to(device, dtype)
          item.data, item.device, item.grad = copy.data, copy.device, None
          moved.append(item)
        elif isinstance(item, Module):
          item.to(device, dtype)
    # realize now: a lazy MLX cast is tied to this thread's stream, and the serving engine runs on another
    for t in moved:
      backend.get(t.device).realize(t.data)
    return self

  def save(self, path):
    """Write all parameters to a .npz file through a temp file and rename, so a crash keeps the old one."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.npz")  # savez appends .npz to a name without it
    np.savez(tmp, **self.state_dict())
    os.replace(tmp, path)

  def load(self, path):
    """Load parameters from a .npz file written by save(); fails if any name or shape differs."""
    with np.load(path) as data:
      state = {k: data[k] for k in data.files}
    expected = {k: v.shape for k, v in self.state_dict().items()}
    got = {k: v.shape for k, v in state.items()}
    assert got == expected, f"{path} does not match this model: {sorted(set(got) ^ set(expected))[:4] or 'shapes differ'}"
    self.load_state_dict(state)
    return self

  def state_dict(self):
    state = {}
    for name, val in self._attrs():
      if isinstance(val, Tensor) and val.requires_grad:
        state[name] = val.numpy()
      elif isinstance(val, np.ndarray):
        # buffers (BatchNorm running stats); without them a reloaded model normalizes with (0, 1)
        state[name] = val.copy()
      elif isinstance(val, Module):
        state.update({f"{name}.{k}": v for k, v in val.state_dict().items()})
    return state

  def load_state_dict(self, state):
    for name, val in self._attrs():
      if isinstance(val, Tensor) and val.requires_grad:
        if name in state:
          assert tuple(state[name].shape) == tuple(val.shape), (name, state[name].shape, val.shape)
          val.data = backend.get(val.device).asarray(state[name]).astype(val.dtype)
      elif isinstance(val, np.ndarray):
        if name in state:
          setattr(self, name, state[name].copy())
      elif isinstance(val, Module):
        child = {k[len(name) + 1:]: v for k, v in state.items() if k.startswith(name + ".")}
        if child:
          val.load_state_dict(child)


class Sequential(Module):
  """Chain modules so the output of each feeds the next."""
  def __init__(self, *layers):
    super().__init__()
    self.layers = list(layers)

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
