# Developer

## Layout

```
stackgrad/
  tensor.py       Tensor: records ops lazily, realize() runs them, backward() walks them in reverse
  mlops.py        the 20 primitives, each with forward, backward, out_shape and out_dtype
  backend/        cpu.py (NumPy), metal.py (MLX), cuda.py (CuPy), and the Metal and Triton kernels
  module.py layers.py conv.py attention.py functional.py loss.py optim.py data.py
  gpt.py qwen3.py tokenizer.py state.py lora.py quant.py sampling.py speculative.py
  serve/          paged KV cache, scheduler, engine, and the FastAPI server
apps/             rag.py and agent.py, clients of the server
train_*.py        TinyStories pretraining, LoRA, GRPO
examples/         GPT-2 and Stable Diffusion v1.5, one file each
tests/            each piece against its reference: PyTorch, transformers, diffusers, finite differences
bench/ scripts/   the measurements; results land in bench/results/
```

## Adding a primitive

A primitive is a `Function` subclass in `mlops.py`. `forward` takes raw backend arrays and saves what the gradient
needs. `backward` takes the output's gradient and returns one gradient per input, `None` where an input needs none.
Because tensors are lazy, the primitive also states its result's shape (`out_shape`) and, if it changes the type,
its dtype (`out_dtype`) before anything runs. `realize` asserts both against the real result.

Composite ops in `tensor.py` must not read `.data`: that runs the graph early. Use `.shape` and `.dtype`, which never
run anything.

## Checks

```sh
uv run python test_grads.py                    # every gradient against finite differences
uv run python tests/parity.py                  # every op and layer against PyTorch
uv run python tests/lazy.py                    # ops record without running, each runs once
uv run python tests/backends.py --device metal # the backend against NumPy
uv run ruff check stackgrad tests bench scripts examples apps train_gpt.py train_lora.py train_grpo.py test_grads.py
```

CI runs these on every push, and fails when the `stackgrad/` package grows past its line budget
(`scripts/lines.py --max`).

## Reproducing the write-up

The post lives at [samitmohan.in/blog/minitorch](https://samitmohan.in/blog/minitorch). Its live figures replay requests
recorded from the server:

```sh
uv run uvicorn stackgrad.serve.api:app --port 8000
uv run python scripts/record_replays.py        # into bench/replays/
```

Each command writes the file its result is read from (`bench/results/`, `runs/`).

```sh
uv run python test_grads.py                    # gradient checks against finite differences
uv run python bench/m1_parity.py --float64     # training beside PyTorch (m1_parity_f64.json)
uv run python bench/lazy.py                    # lazy against eager execution (lazy.json)
uv run python scripts/tinystories.py train     # the tokenizer's training and encoding numbers
uv run python tests/backends.py --device metal # backends against NumPy (backends_metal.json)
uv run python bench/backend_overhead.py        # launch cost, matmul throughput, one training step per device
uv run python bench/flash_attention.py         # the Metal attention kernels against MLX
uv run python bench/profile_step.py --device metal  # where a training step's time goes (profile_metal.json)
uv run python bench/mixed_precision.py --device metal  # mixed precision (mixed_precision_metal.json)
uv run python bench/checkpointing.py --device metal    # activation checkpointing (checkpointing_metal.json)
uv run python scripts/scaling.py fit               # the IsoFLOP fit, from the grid below
uv run python tests/serve.py                   # the scheduler under load (serve_cpu.json)
uv run python tests/qwen3.py                   # Qwen3-0.6B against Hugging Face
uv run python tests/gpt2.py                    # GPT-2 against transformers
uv run python bench/serve_batch.py             # batched serving (serve_batch_metal.json)
uv run python bench/quant.py                   # quantization (quant.json)
uv run python bench/speculative.py             # speculative decoding (speculative_metal.json)
uv run python apps/rag.py                      # retrieval on SQuAD, with the server running
uv run python apps/agent.py                    # the calculator agent, with the server running
uv run python tests/stable_diffusion.py        # Stable Diffusion's networks against diffusers
uv run python bench/stable_diffusion.py        # Stable Diffusion's image beside diffusers'
uv run python bench/gsm8k.py                   # GSM8K accuracy before training

# on an NVIDIA GPU
uv run --no-dev python tests/backends.py --device cuda      # backends_cuda.json
CUPY_TF32=1 uv run --no-dev python scripts/scaling.py run --device cuda   # the scaling-law grid
CUPY_TF32=1 uv run --no-dev python train_gpt.py --run abl_baseline --device cuda --steps 10000 --warmup 500
    # the variants add --norm post, --norm none, --no-rope, or --ffn silu --d-ff 2048; then scripts/ablations.py (ablations.json)
CUPY_TF32=1 uv run --no-dev python bench/mixed_precision.py --device cuda  # mixed_precision_cuda.json
CUPY_TF32=1 uv run --no-dev python bench/checkpointing.py --device cuda --settings off:1,off:2,off:4,off:8,on:1,on:2,on:4,on:8  # checkpointing_cuda.json
CUPY_TF32=1 uv run --no-dev python bench/flash_attention_cuda.py         # flash_attention_cuda.json
CUPY_TF32=1 uv run --no-dev python bench/profile_step.py --device cuda    # profile_cuda.json
CUPY_TF32=1 uv run --no-dev python train_gpt.py --run tinystories --device cuda
CUPY_TF32=1 uv run --no-dev python train_lora.py --run lora_gsm8k --device cuda
CUPY_TF32=1 uv run --no-dev python bench/rollout.py --device cuda          # rollout_cuda.json
CUPY_TF32=1 uv run --no-dev python train_grpo.py --run grpo_gsm8k --device cuda --steps 200
CUPY_TF32=1 uv run --no-dev python train_grpo.py --run grpo_no_stop --device cuda --reward answer --steps 140
uv run python bench/gsm8k.py --adapter runs/lora_gsm8k/adapter.npz   # GSM8K after LoRA
uv run python bench/gsm8k.py --adapter runs/grpo_no_stop/adapter.npz  # after GRPO, reward ignores stopping
uv run python bench/gsm8k.py --adapter runs/grpo_gsm8k/adapter.npz   # after GRPO, reward requires stopping
uv run python bench/gsm8k_compare.py gsm8k_base gsm8k_agent          # the calculator's problem-by-problem split
uv run python bench/gsm8k_compare.py gsm8k_base gsm8k_grpo_gsm8k     # GRPO's split and sign test
uv run python bench/gsm8k_compare.py gsm8k_base gsm8k_lora_gsm8k     # LoRA's split and sign test
```

