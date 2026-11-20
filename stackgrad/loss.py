import math

import numpy as np

from . import functional as F
from .tensor import Tensor


def mse_loss(input, target):
  """Mean squared error between predictions and targets."""
  return ((input - target) ** 2).mean()


def bce_loss(input, target):
  """Binary cross-entropy on probabilities, not logits."""
  p = input.clamp(1e-7, 1 - 1e-7)
  return -(target * p.log() + (1.0 - target) * (1.0 - p).log()).mean()


def cross_entropy_loss(input, target):
  """Softmax cross-entropy over the last axis; `target` holds class indices or one-hot rows."""
  log_probs = F.log_softmax(input, axis=-1)
  labels = target.numpy() if isinstance(target, Tensor) else np.asarray(target)
  if labels.shape == input.shape:
    return -(log_probs * target).sum() / (math.prod(input.shape) // input.shape[-1])
  labels = labels.reshape(-1).astype(np.int64)
  log_probs = log_probs.reshape(-1, input.shape[-1])
  return -log_probs[np.arange(len(labels)), labels].mean()
