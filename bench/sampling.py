import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.gsm8k import PROMPT
from stackgrad import backend, no_grad
from stackgrad.attention import KVCache
from stackgrad.qwen3 import chat_prompt, load_qwen3, load_tokenizer
from train_grpo import load_questions

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cuda", "metal"])
    p.add_argument("--model", default="data/Qwen3-0.6B")
    p.add_argument("--steps", type=int, default=320)
    args = p.parse_args()

    np.random.seed(0)
    tok = load_tokenizer(args.model)
    model = load_qwen3(args.model).to(args.device)
    xp = backend.get(args.device)
    questions = load_questions("data/gsm8k/train.jsonl")[:8]
    prompts = [tok.encode(chat_prompt([{"role": "user", "content": PROMPT.format(question=q)}])) for q, _ in questions for _ in range(8)]
    width = max(len(q) for q in prompts)
    pad = np.array([width - len(q) for q in prompts])
    ids = np.array([[0] * n + q for n, q in zip(pad, prompts)])
    caches = [KVCache() for _ in model.blocks]

    differ = {"first_crossing": 0, "count_below": 0}
    junk = {"first_crossing": 0, "count_below": 0}
    with no_grad():
        logits = model.project(model.hidden(ids, caches, pad)[:, -1])
        for _ in range(args.steps):
            z = logits.data.astype(xp.float32)
            cdf = xp.cumsum(xp.exp(z - z.max(axis=-1, keepdims=True)), axis=-1)
            u = np.random.rand(len(ids), 1).astype(np.float32)
            draw = xp.asarray(u) * cdf[:, -1:]
            picks = {"first_crossing": xp.to_numpy((cdf >= draw).argmax(axis=-1)),
                     "count_below": np.minimum(xp.to_numpy((cdf < draw).sum(axis=-1)), z.shape[-1] - 1)}
            exact_z = xp.to_numpy(z).astype(np.float64)
            prob = np.exp(exact_z - exact_z.max(axis=-1, keepdims=True))
            prob /= prob.sum(axis=-1, keepdims=True)
            exact = (np.cumsum(prob, axis=-1) >= u).argmax(axis=-1)
            for name, got in picks.items():
                off = got != exact
                differ[name] += int(off.sum())
                junk[name] += int((off & (prob[np.arange(len(prob)), got] < 1e-6)).sum())
            logits = model.project(model.hidden(exact[:, None], caches, pad)[:, -1])

    result = {"device": args.device, "draws": args.steps * len(ids), "differ_from_exact": differ, "of_which_below_1e-6": junk}
    print(json.dumps(result, indent=1))
    (RESULTS / f"sampling_{args.device}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
