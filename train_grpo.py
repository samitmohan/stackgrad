import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench.gsm8k import PROMPT, final_number, gold_number, matches
from stackgrad import AdamW, clip_grad_norm
from stackgrad import functional as F
from stackgrad.lora import apply_lora, save_adapter
from stackgrad.qwen3 import chat_prompt, load_qwen3, load_tokenizer
from train_lora import make_batch, reply_logits


def load_questions(path):
  with open(path) as f:
    rows = [json.loads(line) for line in f]
  return [(r["question"], gold_number(r["answer"])) for r in rows]


def reward(text, gold, stopped, require_stop=True):
  """Stopping counts: scored on the final number alone (runs/grpo_no_stop), the model learned by step 51 never to stop."""
  return float((stopped or not require_stop) and matches(final_number(text), gold))


def sample_replies(model, tok, questions, group, max_new, temperature, eos):
  prompts = [tok.encode(chat_prompt([{"role": "user", "content": PROMPT.format(question=q)}])) for q, _ in questions]
  prompts = [p for p in prompts for _ in range(group)]
  width = max(len(p) for p in prompts)
  pad = np.array([width - len(p) for p in prompts])
  ids = np.array([[eos] * n + p for n, p in zip(pad, prompts)])  # any id works as padding: attention hides it
  out = model.generate(ids, max_new, temperature=temperature, eos_id=eos, pad=pad)[:, width:].tolist()
  # a reply keeps its eos, so training also teaches the model when to stop
  replies = [r[:r.index(eos) + 1] if eos in r else r for r in out]
  return prompts, replies


def group_advantages(rewards, group, eps=1e-6):
  r =np.asarray(rewards, dtype=np.float32).reshape(-1, group)
  return ((r - r.mean(axis=1, keepdims=True)) / (r.std(axis=1, keepdims=True) + eps)).reshape(-1)


def policy_loss(model, rows, advantages, n_replies, pad, budget):
  """Divides by the token budget, not the reply's length (Dr. GRPO, Liu et al. 2025); n_replies is the whole step's."""
  inputs, targets, mask = make_batch(rows, pad)
  logits, at = reply_logits(model, inputs, mask)
  logp = F.log_softmax(logits, axis=-1)[np.arange(len(at)), targets.reshape(-1)[at]]
  # `at` holds each reply's tokens side by side, row after row, so each weight repeats once per token
  weight = np.repeat(np.asarray(advantages) / budget / n_replies, mask.sum(axis=1)).astype(np.float32)
  return -(logp * weight).sum()


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--run", required=True)
  p.add_argument("--device", default="cuda", choices=["cpu", "metal", "cuda"])
  p.add_argument("--model", default="data/Qwen3-0.6B")
  p.add_argument("--data", default="data/gsm8k/train.jsonl")
  p.add_argument("--steps", type=int, default=100)
  p.add_argument("--questions", type=int, default=8, help="problems per step")
  p.add_argument("--group", type=int, default=8, help="replies sampled per problem")
  p.add_argument("--max-new", type=int, default=320, help="the evaluation's reply limit (bench/gsm8k.py)")
  p.add_argument("--temperature", type=float, default=1.0)
  p.add_argument("--reward", default="finished", choices=["finished", "answer"],
                 help="finished: the right number in a reply that stops; answer: the right number alone (runs/grpo_no_stop)")
  p.add_argument("--rank", type=int, default=16)
  p.add_argument("--alpha", type=float, default=32)
  # 10x CS336's full fine-tune 1e-5: LoRA trains best near that ("LoRA Without Regret", Schulman et al. 2025)
  p.add_argument("--lr", type=float, default=1e-4)
  # shortcut: one reply per pass, as 2 long replies may not fit on the 3090; measure with block checkpointing before raising
  p.add_argument("--micro", type=int, default=1, help="replies per forward pass in the update")
  p.add_argument("--clip", type=float, default=1.0)
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  np.random.seed(args.seed)
  out = Path("runs") / args.run
  out.mkdir(parents=True, exist_ok=False)
  (out / "config.json").write_text(json.dumps(vars(args), indent=2))
  tok = load_tokenizer(args.model)
  data = load_questions(args.data)
  model = load_qwen3(args.model).to(args.device)
  apply_lora(model, args.rank, args.alpha)
  params = model.parameters()
  opt = AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
  eos, pad = tok.ids[b"<|im_end|>"], tok.ids[b"<|endoftext|>"]
  order = []
  print(f"{args.steps} steps of {args.questions} problems x {args.group} replies -> {out}", flush=True)

  start = time.time()
  with open(out / "metrics.jsonl", "w") as metrics, open(out / "samples.jsonl", "w") as samples:
    for step in range(1, args.steps + 1):
      if len(order) < args.questions:
        order = list(np.random.permutation(len(data)))
      questions = [data[order.pop()] for _ in range(args.questions)]
      t0 = time.perf_counter()
      prompts, replies = sample_replies(model, tok, questions, args.group, args.max_new, args.temperature, eos)
      t1 = time.perf_counter()
      texts = [tok.decode([t for t in r if t != eos]) for r in replies]
      golds = np.repeat([gold for _, gold in questions], args.group)
      rewards = [reward(text, gold, r[-1] == eos, args.reward == "finished") for text, gold, r in zip(texts, golds, replies)]
      adv = group_advantages(rewards, args.group)

      # zero-advantage replies add no gradient, so skip them; sort by length so micro-batches pad less
      keep = sorted(np.flatnonzero(adv), key=lambda i: len(prompts[i]) + len(replies[i]))
      opt.zero_grad()
      for j in range(0, len(keep), args.micro):
        chunk = keep[j:j + args.micro]
        loss = policy_loss(model, [(prompts[i] + replies[i], len(prompts[i])) for i in chunk], adv[chunk], len(adv), pad, args.max_new)
        loss.backward()
        del loss  # free this graph's saved activations before the next forward
      grad_norm = 0.0
      if keep:  # no gradient, no step: Adam would still count one
        grad_norm = clip_grad_norm(params, args.clip)
        opt.step()
      t2 = time.perf_counter()

      reply_tokens = sum(len(r) for r in replies)
      by_group = np.reshape(rewards, (-1, args.group))
      row = {"step": step, "reward": round(float(np.mean(rewards)), 4),
             "groups_with_signal": round(float(np.mean(by_group.max(axis=1) > by_group.min(axis=1))), 3),
             "reply_tokens": round(reply_tokens / len(replies), 1),
             "hit_limit": round(float(np.mean([r[-1] != eos for r in replies])), 3),
             "grad_norm": round(float(grad_norm), 4), "rollout_s": round(t1 - t0, 1), "update_s": round(t2 - t1, 1),
             "rollout_tokens_per_s": round(reply_tokens / (t1 - t0)), "time": round(time.time() - start, 1)}
      metrics.write(json.dumps(row) + "\n")
      metrics.flush()
      samples.write(json.dumps({"step": step, "question": questions[0][0], "text": texts[0], "reward": rewards[0]}) + "\n")
      samples.flush()
      print(f"step {step:4d}/{args.steps} | reward {row['reward']:.3f} | reply {row['reply_tokens']:.0f} tok | "
            f"rollout {row['rollout_s']:.0f} s ({row['rollout_tokens_per_s']:,} tok/s) | update {row['update_s']:.0f} s", flush=True)
      if step % 10 == 0 or step == args.steps:
        save_adapter(model, out / "adapter.npz", args.rank, args.alpha)


if __name__ == "__main__":
  main()
