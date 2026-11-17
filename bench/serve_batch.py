"""Serving N chats at once on the Apple GPU: one forward pass per sequence against one per step.

    uv run python bench/serve_batch.py

Qwen3-0.6B in bfloat16 behind the Scheduler and a 512-block KV budget, as minitorch/serve/api.py runs
it. N GSM8K prompts in the chat template, of different lengths, arrive together, and each decodes 128
tokens greedily with no stop token. The runner's decode as it was before batching, one forward pass
per sequence, runs beside the batched one (runner.py). Reports the time for all N replies, the tokens
per second they add up to, the median decode step, and how many replies match the unbatched ones
token for token (bfloat16 rounds a padded batch differently, so a near-tie can flip). Then the same
match in float32, where only rounding error separates the two. Writes bench/results/serve_batch_metal.json.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.gsm8k import PROMPT
from minitorch import no_grad
from minitorch.qwen3 import chat_prompt, load_qwen3, load_tokenizer
from minitorch.serve.kv_cache import BlockManager
from minitorch.serve.runner import MinitorchRunner
from minitorch.serve.scheduler import Request, Scheduler

RESULTS = Path(__file__).resolve().parent / "results"
MAX_NEW = 128
REPEATS = 3


class OnePerSequence(MinitorchRunner):
    """The runner before batching: each sequence's decode is its own forward pass."""
    def decode(self, seq_ids, tokens):
        with no_grad():
            return np.stack([self.model(np.array([[t]]), self.caches[s])[0, -1].numpy() for s, t in zip(seq_ids, tokens)])


def serve(runner, prompts):
    """Every prompt arrives at once. Returns each step's seconds (the first prefills them all) and each reply."""
    s = Scheduler(runner, BlockManager(512, 16))
    requests = [Request(i, ids, MAX_NEW, temperature=0) for i, ids in enumerate(prompts)]
    for r in requests:
        s.add(r)
    steps = []
    while s.has_work():
        start = time.perf_counter()
        s.step()
        steps.append(time.perf_counter() - start)
    return steps, [r.out for r in requests]


def main():
    tok = load_tokenizer("data/Qwen3-0.6B")
    model = load_qwen3("data/Qwen3-0.6B").to("metal", "bfloat16")
    with open("data/gsm8k/test.jsonl") as f:
        questions = [json.loads(line)["question"] for line in f][:8]
    prompts = [tok.encode(chat_prompt([{"role": "user", "content": PROMPT.format(question=q)}])) for q in questions]
    modes = {"one forward per sequence": OnePerSequence, "one forward per step": MinitorchRunner}
    for runner in modes.values():
        serve(runner(model), prompts[:2])  # warm up: the first calls allocate

    rows = []
    for n in (1, 2, 4, 8):
        replies = {}
        for name, runner in modes.items():
            runs = [serve(runner(model), prompts[:n]) for _ in range(REPEATS)]
            seconds = statistics.median(sum(steps) for steps, _ in runs)
            replies[name] = runs[0][1]
            rows.append({"mode": name, "concurrent": n, "seconds": round(seconds, 3),
                         "tokens_per_s": round(n * MAX_NEW / seconds, 1),
                         "decode_step_ms": round(1000 * statistics.median(t for steps, _ in runs for t in steps[1:]), 2)})
        rows[-1]["identical_replies"] = sum(a == b for a, b in zip(*replies.values()))
        for r in rows[-2:]:
            print(r, flush=True)

    del model
    model = load_qwen3("data/Qwen3-0.6B").to("metal", "float32")
    replies = [serve(runner(model), prompts)[1] for runner in modes.values()]
    float32_identical = sum(a == b for a, b in zip(*replies))
    print(f"float32: {float32_identical} of {len(prompts)} replies identical", flush=True)

    result = {"device": "Apple M4 GPU", "model": "Qwen3-0.6B", "dtype": "bfloat16", "new_tokens": MAX_NEW,
              "prompt_tokens": [len(p) for p in prompts], "repeats": REPEATS, "rows": rows,
              "float32_identical_replies": float32_identical, "float32_requests": len(prompts)}
    (RESULTS / "serve_batch_metal.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
