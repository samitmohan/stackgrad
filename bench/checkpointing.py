import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stackgrad.lora import apply_lora
from stackgrad.qwen3 import load_qwen3, load_tokenizer
from train_lora import examples, make_batch, reply_loss

RESULTS = Path(__file__).resolve().parent / "results"


def peak_tracker(device):
  if device == "metal":
    import mlx.core as mx
    return mx.reset_peak_memory, mx.get_peak_memory
  import cupy as cp
  pool = cp.get_default_memory_pool()
  # after free_all_blocks the pool only grows to the largest demand, so its size is the peak
  return pool.free_all_blocks, pool.total_bytes


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--device", default="metal", choices=["metal", "cuda"])
  p.add_argument("--settings", default="off:1,on:1,on:2,on:4,on:8", help="checkpointing:examples per pass, comma separated")
  args = p.parse_args()

  np.random.seed(0)
  tok = load_tokenizer("data/Qwen3-0.6B")
  batch = sorted(examples(tok, "data/gsm8k/train.jsonl")[:8], key=lambda r: len(r[0]))
  model = load_qwen3("data/Qwen3-0.6B").to(args.device)
  apply_lora(model, 16, 32)
  pad = tok.ids[b"<|endoftext|>"]
  reset, read = peak_tracker(args.device)
  reply_tokens = sum(len(ids) - n for ids, n in batch)

  def update(micro):
    for p in model.parameters():
      p.grad = None
    for i in range(0, len(batch), micro):
      inputs, targets, mask = make_batch(batch[i:i + micro], pad)
      loss = reply_loss(model, inputs, targets, mask) * (mask.sum() / reply_tokens)
      loss.backward()
      del loss
    model.parameters()[0].grad.sum().item()  # wait for the device

  rows = []
  for setting in args.settings.split(","):
    mode, micro = setting.split(":")
    model.checkpoint_blocks = mode == "on"
    reset()
    try:
      update(int(micro))  # once to warm up, then timed
      start = time.perf_counter()
      update(int(micro))
      rows.append({"checkpointing": mode == "on", "micro": int(micro), "peak_gb": round(read() / 1e9, 2),
                   "seconds_per_batch": round(time.perf_counter() - start, 2)})
    except Exception as e:  # CuPy raises OutOfMemoryError
      if "memory" not in str(e).lower():
        raise
      rows.append({"checkpointing": mode == "on", "micro": int(micro), "out_of_memory": True})
    print(rows[-1], flush=True)

  result = {"device": args.device, "examples": len(batch), "tokens": int(sum(len(ids) for ids, _ in batch)), "rows": rows}
  (RESULTS / f"checkpointing_{args.device}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
  main()
