from importlib.metadata import version

HELP = """stackgrad {}: a deep learning framework, from autograd to a served Qwen3 with retrieval and an agent.

  import numpy as np
  from stackgrad import Tensor
  w = Tensor(np.random.randn(3, 1), requires_grad=True)
  ((Tensor(np.random.randn(8, 3)) @ w) ** 2).mean().backward()
  print(w.grad)

GPUs: pip install "stackgrad[metal]" (Apple) or "stackgrad[cuda]" (NVIDIA). Chat server: "stackgrad[serve]".

  docs        https://samitmohan.github.io/stackgrad/
  quickstart  https://samitmohan.github.io/stackgrad/quickstart/
  code        https://github.com/samitmohan/stackgrad
  write-up    https://samitmohan.in/blog/minitorch"""


def main():
  print(HELP.format(version("stackgrad")))


if __name__ == "__main__":
  main()
