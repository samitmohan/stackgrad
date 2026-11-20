import functools
import importlib

DEVICES = ("cpu", "metal", "cuda")


@functools.cache
def get(device):
  if device not in DEVICES:
    raise ValueError(f"unknown device {device!r}, expected one of {DEVICES}")
  return importlib.import_module(f"{__name__}.{device}")
