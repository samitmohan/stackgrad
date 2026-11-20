import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stackgrad import TransformerLM, autocast, cross_entropy_loss
from stackgrad.lora import apply_lora

CONFIG = {"vocab_size": 50, "context_length": 64, "d_model": 32, "n_layers": 3, "n_heads": 4, "d_ff": 64, "n_kv_heads": 2}


def grads(model, idx, checkpointing, cast=None):
    model.checkpoint_blocks = checkpointing
    for p in model.parameters():
        p.grad = None
    with autocast(cast):  # None: every multiply in float32
        loss = cross_entropy_loss(model(idx), (idx + 1) % CONFIG["vocab_size"])
    loss.backward()  # outside autocast, as train_gpt.py --amp runs it, so the recompute must restore it
    return loss.item(), {n: p.grad for n, p in model.named_parameters()}


def check(model, idx, what, cast=None):
    loss0, g0 = grads(model, idx, False, cast)
    loss1, g1 = grads(model, idx, True, cast)
    gap = max(float(np.abs(g0[n] - g1[n]).max()) for n in g0)
    assert loss0 == loss1 and gap < 1e-6 and all(g is not None for g in g1.values()), (loss0, loss1, gap)
    print(f"[ok]   {what}: the same loss and {len(g0)} gradients with checkpointing (largest gap {gap:.1e})")


def main():
    np.random.seed(0)
    idx = np.random.randint(0, CONFIG["vocab_size"], (2, 24))
    check(TransformerLM(**CONFIG), idx, "full training")
    lora = TransformerLM(**CONFIG)
    apply_lora(lora, r=4, alpha=8)
    check(lora, idx, "LoRA, where no block input needs a gradient but the adapters inside do")
    check(TransformerLM(**CONFIG), idx, "float16 autocast", cast="float16")


if __name__ == "__main__":
    main()
