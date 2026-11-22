import argparse
import json
import sys
import time
from pathlib import Path

import cupy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stackgrad.qwen3 import load_qwen3, load_tokenizer
from train_grpo import load_questions, reward, sample_replies


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", required=True, help="kept replies as GSM8K-style jsonl, for train_lora.py --data")
  p.add_argument("--model", default="data/Qwen3-0.6B")
  p.add_argument("--data", default="data/gsm8k/train.jsonl")
  p.add_argument("--samples", type=int, default=3, help="replies sampled per problem; 32 x 4 = 128 rows ran out of memory")
  p.add_argument("--questions", type=int, default=32, help="problems per generate call")
  p.add_argument("--limit", type=int, default=None, help="first N problems only, for a throughput test")
  p.add_argument("--max-new", type=int, default=320, help="the evaluation's reply limit (bench/gsm8k.py)")
  p.add_argument("--temperature", type=float, default=0.8)
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  np.random.seed(args.seed)
  data = load_questions(args.data)[:args.limit]
  tok = load_tokenizer(args.model)
  model = load_qwen3(args.model).to("cuda")
  # shortcut: realize() keeps every intermediate of the graph it runs alive until it returns, so a whole-model prefill of
  # 16 rows x 96 tokens took 12.3 GB on the 3090 vs 3.1 GB realized block by block; drop this once realize frees them
  for blk in model.blocks:
    blk.forward = lambda *a, f=blk.forward: f(*a).realize()
  eos = tok.ids[b"<|im_end|>"]
  sampled, correct, kept_tokens, solved, peak, start = 0, 0, [], 0, 0.0, time.time()
  pool = cupy.get_default_memory_pool()
  with open(args.out, "w") as out:
    for b in range(0, len(data), args.questions):
      questions = data[b:b + args.questions]
      _, replies = sample_replies(model, tok, questions, args.samples, args.max_new, args.temperature, eos)
      for qi in range(len(questions)):
        seen = set()
        for r in replies[qi * args.samples:(qi + 1) * args.samples]:
          text = tok.decode([t for t in r if t != eos])
          # the GRPO reward: the right final number in a reply that stops
          ok = reward(text, questions[qi][1], r[-1] == eos)
          correct += ok
          if ok and text not in seen:
            seen.add(text)
            kept_tokens.append(len(r))
            out.write(json.dumps({"question": questions[qi][0], "answer": text}) + "\n")
        solved += bool(seen)
      sampled += len(replies)
      out.flush()
      peak = max(peak, pool.total_bytes() / 1e9)  # the pool keeps freed blocks, so this is the most the batch held
      # shortcut: the growing KV caches fragment the pool even so, and 96 rows ran out of memory after 3,328 problems; a
      # preallocated KV cache in stackgrad fixes it, and until then a --start offset would let a crashed run resume
      pool.free_all_blocks()
      dt = time.time() - start
      print(f"{b + len(questions):5d}/{len(data)} problems | {correct / sampled:.3f} correct | {len(kept_tokens)} kept | peak {peak:.1f} GB | "
            f"{dt:.0f} s, {sampled / dt:.2f} replies/s, eta {dt / (b + len(questions)) * (len(data) - b - len(questions)):.0f} s", flush=True)

  stats = vars(args) | {"problems": len(data), "replies": sampled, "correct": int(correct), "kept": len(kept_tokens),
                        "problems_with_kept": solved, "kept_reply_tokens": round(float(np.mean(kept_tokens)), 1),
                        "peak_pool_gb": round(peak, 2), "seconds": round(time.time() - start, 1)}
  Path(args.out).with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))
  print(json.dumps(stats, indent=1))


if __name__ == "__main__":
  main()
