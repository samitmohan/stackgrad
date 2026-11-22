# Interview prep

This page prepares you to defend stackgrad in an interview for an ML infra, inference, ML engineer, or research engineer
role. An interviewer can pick any layer of the project and keep digging, so the page walks every layer in the order the
post tells it, with the code that implements it, the number it measured, and the questions you will get.

## How to use this page

Every answer is collapsed. Read the question, answer it out loud as if the interviewer were in the room, then open the
answer and compare. Answers are written the way you would say them, in the first person, with the numbers in.

Each area has four parts:

- what it is and the one design decision that matters, intuition first;
- code pointers as `path:line`, all checked against the current tree;
- the measured numbers, each with the file it comes from;
- questions, mixing basic, deep, and "why not X", with at least one whiteboard exercise worked out.

A number on this page comes from `bench/results/*.json`, `runs/*`, or the post (`drafts/karpathy/post.md`). Arithmetic
you can redo on a whiteboard says "arithmetic". The debugging stories quote a few numbers that live only in the
2026-11-21 session log or in a code comment, and each says so where it appears. When an interviewer asks for a number
this page lacks, say it needs measuring.

Pair this page with the [Study plan](guide.md): the plan rebuilds each piece, this page rehearses explaining it.
Tick a section's checklist when you can do every item without notes.

## The pitch

??? question "The 30-second version"
    stackgrad is a deep learning framework I wrote from scratch in under 3,000 lines of Python. It has reverse-mode
    autograd over 19 gradient operations, lazy evaluation, and three backends: NumPy, MLX on the Apple GPU, and CuPy on an
    NVIDIA GPU, with my own Metal and Triton attention kernels. On top of it I load Qwen3-0.6B from its published
    weights and match Hugging Face's logits to 5.1e-5, serve it with a KV cache and continuous batching, and fine-tune it
    with LoRA, GRPO, and rejection-sampling fine-tuning. My favorite systems result: recording the decode step as one
    compiled graph took Qwen3's layers from 25.5 ms to 11.9 ms per token, so decoding runs at 82% of the laptop's memory
    bandwidth.

??? question "The 2-minute version"
    I wanted to know every piece of a modern LLM stack well enough to write it, so I did, in one codebase.

    It starts with autograd. Every gradient lives in one file, `mlops.py`, in 19 operations, and everything else
    composes them. Trained side by side with PyTorch in float64, the losses agree to 1.8e-15 at step 1. Tensors are lazy
    like tinygrad's: an op records itself and runs when you read the result.

    Then a transformer in the modern form CS336 teaches: RMSNorm, RoPE, SwiGLU, grouped-query attention, and my own
    byte-level BPE tokenizer. I trained a 22.7M-parameter model on TinyStories to a validation loss of 1.399 in 71
    minutes on one RTX 3090, at 23.9% MFU, and ran the CS336 ablations.

    For speed I wrote attention kernels. My Metal kernel runs 6.7x faster than MLX's built-in attention on the story
    model's narrow heads, and my Triton FlashAttention-2 forward plus backward runs 8.5x faster than composed attention.

    Then a real model: I parse safetensors, map Qwen3's tensor names, and match Hugging Face to 5.1e-5 in float32. I
    serve it with a block manager, continuous batching, and preemption: 8 chats in one batched pass write 155 tokens per
    second against 63 for one pass per chat. Graph capture with a fixed-capacity KV cache made a decode step 2.15x
    faster with identical tokens.

    Last, I tried to make it better at grade-school math. LoRA made it worse, 152 to 123 of 250, a real drop. GRPO's
    first reward got gamed: the model learned to repeat its answer to the token limit. With a fixed reward, GRPO and RFT
    both landed inside the noise of a paired sign test. I published those results as measured, null results included.

### Five numbers to know cold

| Number | What it measures | Source |
|---|---|---|
| 1.8e-15 | stackgrad vs PyTorch loss at step 1, same model, float64 | `bench/results/m1_parity_f64.json` |
| 5.1e-5 | Largest logit gap vs Hugging Face's Qwen3-0.6B, CPU float32 (largest logit 24.3) | `bench/results/qwen3_parity.json` |
| 1.399 in 71 min, 23.9% MFU | TinyStories validation loss, 40,000 steps on one RTX 3090 | `runs/tinystories/metrics.jsonl` |
| 25.5 to 11.9 ms | Qwen3-0.6B decode step through the layers, capture off vs on (2.15x) | `bench/results/compile_metal.json` |
| 152 of 250 | Released Qwen3-0.6B on the fixed GSM8K subset; LoRA 123 (p = 0.0011), GRPO 139 (p = 0.13), RFT 155 (p = 0.73) | `bench/results/gsm8k_*.json` |

- [ ] I can give the 30-second pitch in 30 seconds.
- [ ] I can give the 2-minute pitch without notes and stop at any sentence to go deeper.
- [ ] I know the five numbers and where each came from.

## Architecture map

Start with one token's path. Text becomes ids, ids go through the model, and every op the model calls records itself
instead of running:

```text
text
  |  Tokenizer.encode                         stackgrad/tokenizer.py
  v
ids (B, T)
  |  TransformerLM.hidden / project           stackgrad/gpt.py
  v
Embedding -> Block x L -> RMSNorm -> head -> logits
  |  each op calls _apply(): records, runs nothing     stackgrad/tensor.py:56
  v
lazy graph of mlops Functions                 stackgrad/mlops.py
  |  reading .data / .numpy() / .backward() calls Tensor.realize()   stackgrad/tensor.py:149
  v
backend module xp: numpy | mlx | cupy         stackgrad/backend/{cpu,metal,cuda}.py
  |
  v
kernels: BLAS, MLX, CuPy, plus mine:
  Metal attention v1 and v3                   stackgrad/backend/metal_attention.py
  Metal int4/int8 matmul                      stackgrad/backend/metal_quant.py
  Triton FlashAttention-2, forward + backward stackgrad/backend/cuda_attention.py
```

Now add who calls the model. Serving wraps it in a scheduler, training scripts wrap it in an optimizer loop, and the
apps talk to the server over HTTP:

```text
  apps/rag.py     apps/agent.py
        \           /
   HTTP /v1/chat/completions
             |
  serve/api.py -> Engine (thread) -> Scheduler -> BlockManager
                                        |
                                 StackgradRunner (prefill one, decode the batch)
                                        |
  qwen3.load_qwen3 (safetensors) -> TransformerLM <- lora.apply_lora / merge_adapter
                                        |
                       Tensor -> realize -> mlops -> backend
                                        ^
  train_gpt.py, train_lora.py, train_grpo.py: AdamW, cosine_lr, clip_grad_norm
  scripts/rft_sample.py -> kept replies -> train_lora.py --data
```

- [ ] I can draw both diagrams from memory and name the file behind each box.

## Autograd and the 19 ops

A tensor remembers the operation that produced it. Calling `backward()` on the loss sorts the graph so every tensor
comes after its inputs, then walks it from the loss back, and each operation multiplies the incoming gradient by its own
local slope. The design decision is tinygrad's: a short list of primitives with hand-written gradients, and everything
else composed from them, so a new layer gets its gradient for free.

`mlops.py` holds 21 `Function` classes. Nineteen carry gradients. `Less` is marked non-differentiable, and `Write` (the
KV cache write, added 2026-11-21) has no backward because caches only fill under `no_grad`. Three primitives exist for a
reason other than minimality: `Sigmoid` (composed from exp and recip, a very negative input gives `0 * inf = nan` in
backward), `Matmul` (as broadcast-multiply-sum it would lose BLAS), and `Attention` (so a backend can swap in a fused
kernel).

Code:

- `stackgrad/tensor.py:56-71`: `_apply` records an op and decides `requires_grad`.
- `stackgrad/tensor.py:74-78`: `_broadcast` expands only inputs that need a gradient.
- `stackgrad/tensor.py:343-373`: `backward`, iterative topological sort, then the reverse walk.
- `stackgrad/mlops.py:37-43`: `unbroadcast`, the sum that undoes broadcasting.
- `stackgrad/mlops.py:114-121`: `Mul`; `stackgrad/mlops.py:282-298`: `Matmul`.
- `stackgrad/functional.py:26-36`: softmax and log-softmax with a detached max.
- `stackgrad/tensor.py:9-26`: `no_grad`, thread-local.

Numbers:

- Losses agree to 1.8e-15 at step 1 in float64 and within 8.4e-4 over 200 steps (`bench/results/m1_parity_f64.json`).
- 52 of 52 ops agree with NumPy on Metal (`bench/results/backends_metal.json`).

??? question "Walk me through what happens when you call loss.backward()."
    First it checks the loss is a scalar and that something in the graph requires a gradient
    (`stackgrad/tensor.py:345-346`). Then it does an iterative depth-first search from the loss over parents that need
    gradients, appending each node after its inputs, which gives a topological order (`tensor.py:349-357`). It's
    iterative so a deep graph can't hit Python's recursion limit. It seeds the loss's gradient with ones and walks the
    order in reverse. Each node hands its gradient to its Function's `backward`, gets one gradient per parent, and adds
    it to that parent's `.grad`. Intermediate grads are freed once passed on, unless you ask to keep them.

??? question "Whiteboard: derive Mul's backward, then Matmul's."
    For `z = x * y` elementwise, `dz/dx = y` and `dz/dy = x`. If the gradient arriving at z is g, the chain rule gives
    `g * y` for x and `g * x` for y. That's `stackgrad/mlops.py:119-121`, and it skips a side that doesn't need a gradient.

    For `Z = X @ W` with X (m, k), W (k, n), Z (m, n): each `Z[i,j] = sum_k X[i,k] W[k,j]`, so
    `dL/dX[i,k] = sum_j G[i,j] W[k,j]`, which is `G @ W.T`, shape (m, n)(n, k) = (m, k). And
    `dL/dW = X.T @ G`, (k, m)(m, n) = (k, n). Check: the gradient always has its input's shape. Batched inputs then get
    `unbroadcast` back to their own shape (`mlops.py:287-289`). One forward matmul costs two backward matmuls, which is
    why a backward pass costs about twice a forward.

??? question "Why does backward add gradients with `parent.grad + grad` and never `+=`?"
    `Add.backward` returns the same gradient array to both parents (`mlops.py:110-111`). An in-place `+=` on one parent's
    grad would mutate the array the other parent also holds. The comment at `tensor.py:372` says so. A tensor used in
    two places needs the sum of both incoming gradients, which the out-of-place add gives.

??? question "How do you handle broadcasting in the backward pass?"
    Two pieces. Before a binary op, `_broadcast` expands only the inputs that require a gradient to the output shape
    (`tensor.py:74-78`). Constants broadcast for free inside NumPy. Expand's backward is `unbroadcast`: sum away the
    leading dims that were added, then sum with keepdims over every axis that was size 1 (`mlops.py:37-43`). Intuition:
    if one bias value fed 32 rows, it collects the 32 rows' gradients.

??? question "Why is Sigmoid a primitive when it's 1 / (1 + exp(-x))?"
    Composed from Exp and Recip, a very negative x makes `exp(-x)` overflow to inf, and the backward multiplies
    something by that inf and something by zero, giving `0 * inf = nan`. As one primitive the backward is
    `grad * s * (1 - s)` on the saved output, which stays finite (`mlops.py:73-80`). The CPU forward also silences the
    harmless overflow warning, since `1 / inf = 0` is the right answer (`backend/cpu.py:84-87`).

??? question "Max with ties: where does the gradient go?"
    It splits in equal shares. The backward builds a mask of positions equal to the max and divides by the count
    (`mlops.py:170-176`). Any split that sums to the incoming gradient is a valid subgradient. An even split matches what
    a finite-difference check sees when tied inputs move together.

??? question "Why doesn't softmax's subtracted max need a gradient?"
    Softmax is shift-invariant: `softmax(x - c) = softmax(x)` for any constant c, so the output's derivative with respect
    to c is zero. I `detach()` the max (`functional.py:29`), which saves building and walking a Max backward for a term
    that contributes nothing.

??? question "How do you know your gradients are right?"
    Two independent checks. Finite differences: nudge each input by h = 1e-5 up and down and compare the slope with
    `backward()` (`test_grads.py`). And PyTorch: the same 8.3M-parameter model, same weights and batches, trained 200
    steps in both. In float64 they agree to 1.8e-15 at step 1. One lesson from the checks: the LayerNorm test first summed
    the output, and for a plain sum LayerNorm's true input gradient is zero, so a wrong gradient passed.
    Multiplying by random weights first made it a real test.

??? question "What does the new requires_grad assertion in backward catch?"
    `backward()` on a tensor whose graph has nothing that requires a gradient now raises a clear error
    (`tensor.py:346`). Before, it walked an empty graph and filled nothing, so you'd find out later when `.grad` was
    None. I checked it: `(Tensor([1.0, 2.0]) * 2).sum().backward()` raises "nothing this tensor depends on has
    requires_grad=True". One side effect to know: activation checkpointing around a block with no trainable weights and
    an input without grad now raises too, since `checkpoint` marks its output as requiring grad
    (`checkpoint.py:36-37`). No training script hits it, because LoRA puts trainable weights in every checkpointed block.

??? question "Why is no_grad thread-local?"
    It began as one global flag. While the chat engine's thread sat inside `no_grad`, another request thread built a
    model, which came out with `requires_grad=False` everywhere and stayed cached with frozen random weights. Now the
    flag lives in a `threading.local()` (`tensor.py:9`). Anything global in a server breaks on the second request.

??? question "Why not use PyTorch's autograd, or JAX?"
    The project's goal was writing every piece, and PyTorch, tiktoken and transformers are test oracles only, never
    imported from `stackgrad/`. Writing autograd made me answer questions a framework hides: where the max goes in
    softmax, why `+=` is a bug, what Expand's backward is. The cost is speed: on CPU float32 stackgrad takes 191 ms per
    step against PyTorch's 90 ms on the same model (`bench/results/m1_parity.json`).

- [ ] I can derive Mul, Matmul, Sum, Expand, and Max backward on a whiteboard.
- [ ] I can write `backward()`'s topological walk from memory.
- [ ] I can explain why Sigmoid, Matmul, and Attention are primitives.

## Laziness and realize

Typing `c = a @ b` records the multiplication and returns a tensor that knows its shape and dtype but holds no numbers.
Reading it runs every op it depends on, inputs first, once each. Laziness lets `realize` see a whole chain before
anything runs. Since 2026-11-21 it uses that view twice: it runs chains of elementwise ops as one fused call, and it
frees each buffer as soon as its last reader has run.

The design decision for fusion is the shortcut over tinygrad's compiler. tinygrad schedules the graph into kernels and
generates Metal or CUDA source. stackgrad finds runs of consecutive elementwise nodes and hands each run to the backend's
`fuse`, which is `mx.compile` on Metal and the identity on CPU and CUDA. All math still lives in `mlops.py`.

A node joins a run when four things hold (`stackgrad/tensor.py:85-88`): autograd isn't recording it (`_ctx is None`),
the profiler is off, its op is in `_FUSABLE` (elementwise ops plus Expand and Reshape), and its output has at least
2^16 = 65,536 elements. The fused function is cached by the run's structure (op types, kwargs, which inputs come from
inside the run, dtypes), so a decode step that rebuilds the same graph reuses one compiled function.

Code:

- `stackgrad/tensor.py:56-71`: `_apply` builds `_lazy = (ctx, kwargs, inputs)` with the output shape and dtype.
- `stackgrad/mlops.py:20-29`: default `out_shape` (broadcast) and `out_dtype` (promotion probe).
- `stackgrad/tensor.py:149-184`: `realize`: post-order DFS, runs, `needed`, freeing, execution.
- `stackgrad/tensor.py:81-124`: `_FUSABLE`, `_fusable`, `_store`, `_realize_fused`, `_fused` (lru_cache).
- `stackgrad/backend/metal.py:85`: `fuse = mx.compile`; `stackgrad/backend/cpu.py:79-81`: identity;
  `stackgrad/backend/cuda.py:24`: identity, marked as a shortcut.
- `tests/lazy.py:90-131`: fused equals unfused, the structure cache hits, a training step is bitwise equal.

Numbers:

- Laziness alone: 4.8 ms vs 4.09 ms per step on Metal, 10.04 vs 10.01 ms on CPU, identical losses
  (`bench/results/lazy.json`).
- Fusion: 512-token Qwen3 prefill 216.98 to 213.24 ms (1.02x); padded 8-row decode 36.82 to 36.56 ms (1.01x); outputs
  identical (`bench/results/compile_metal.json`).
- Freeing intermediates: on Metal, 3.71 GB peak either way (`bench/results/realize_memory_metal.json`). On CUDA, 6.85 GB
  to 0.19 GB above the weights for a 16 x 96 prefill (2026-11-21 session log; no CUDA result file was saved).

??? question "What does lazy mean in stackgrad, and what triggers execution?"
    `_apply` builds the op's context, computes the output shape and dtype from the inputs' shapes and dtypes, and returns
    a tensor with `_buf = None` and `_lazy` set (`tensor.py:56-71`). Nothing runs. Reading `.data`, `.numpy()`,
    `.item()`, printing, `.backward()`, or `.realize()` runs the graph (`tensor.py:138-143`). An op nobody reads never
    runs.

??? question "Why does every primitive declare out_shape and out_dtype?"
    The next op is recorded before this one runs, so it needs this op's shape and dtype from metadata alone. Each
    Function states them, and `_store` asserts that the computed array matches what was recorded
    (`tensor.py:91-97`). Index is the interesting one: it indexes a zero-stride stand-in array of the input's shape,
    so NumPy's own indexing rules compute the result shape without touching data (`mlops.py:230-240`). A related rule:
    composite ops must read `.shape` and `.dtype`, never `.data`, or they run the graph early. `mean` computes its scale
    from shapes for that reason (`tensor.py:257-261`).

??? question "Walk me through realize's fusion."
    Step one, a post-order DFS gives every unrealized node after its inputs (`tensor.py:151-159`). Step two, walk that
    order and group consecutive fusable nodes into runs; anything else is a run of one (`tensor.py:160-165`). Step three,
    mark which nodes are needed outside their run: the caller's tensor, plus any input read by a node in a different run
    (`tensor.py:167-169`). Step four, execute: a run of two or more goes through `_realize_fused`, which turns the run
    into a structural key, fetches the compiled function from an LRU cache, calls it with the external inputs, and
    stores only the needed outputs (`tensor.py:101-124`). A run of one calls `forward` on its own.

??? question "What happens to an intermediate a fused run computed but didn't return?"
    It keeps its `_lazy` record. If you read it later, realize recomputes it, bit for bit. `tests/lazy.py:107-110`
    checks that. The trade is deliberate: returning every intermediate would write them all to memory, which is
    the traffic fusion exists to remove.

??? question "Whiteboard: why is fusion skipped below 2^16 elements?"
    Fusion saves memory passes. For `y = (x * 2 + 1).relu()` eager runs three kernels that each read N and write N, 6N
    numbers moved. Fused, it reads N and writes N, 2N moved.

    Now price one pass at the threshold (arithmetic). 65,536 float32 values is 256 KiB; reading and writing them moves
    512 KiB, which at the measured 101.4 GB/s (`bench/results/quant.json`) takes about 5.2 microseconds. One elementwise
    op through stackgrad on Metal costs 7.59 microseconds even at 1,000 elements (`bench/results/backend_overhead_cpu_metal.json`),
    and the fused path adds Python of its own: building the key, the cache lookup, the call. Below that size you'd pay
    more Python than the bytes you save. A decode step's activations are 8 rows of 1,024 values, 8,192 elements, far
    under the line, and the padded decode row confirms it: 1.01x. I chose 2^16 by this reasoning; no sweep of the
    threshold exists in the results.

??? question "Why does fusion only give 1.02x on prefill?"
    Prefill is matmul time, and fusion doesn't touch matmuls. Arithmetic: Qwen3-0.6B has 440.5M weights in its blocks
    (596.0M minus the 155.6M embedding table, which prefill uses as a lookup). A 512-token prefill does
    2 x 440.5M x 512 = 451 GFLOP of matmul. At the Apple GPU's measured 3.62 TFLOP/s in bfloat16
    (`bench/results/backend_overhead_cpu_metal.json`) that's 125 ms even at peak, of the 213 ms measured. Fusion can only
    work on the elementwise remainder, and it recovered about 4 ms of it.

??? question "Why can't training fuse?"
    A node autograd records keeps its forward's arrays for the backward: Mul saves x and y, Exp saves its output. A
    compiled trace returns only its outputs, so it can't hand those saved arrays back. `_fusable` requires
    `node._ctx is None` (`tensor.py:88`). `tests/lazy.py:113-131` shows the boundary: inside a training step, a sub-chain
    that needs no gradient does fuse, and the loss and gradients stay bitwise equal.

??? question "Realize now frees intermediates. What held them before, and why did it matter on CUDA but not Metal?"
    realize's own bookkeeping: the `order` list referenced every node of the graph until it returned, so every buffer
    stayed alive. Now it deletes `order`, `seen`, and `run_of` before executing, sets `runs[i] = None` as each run
    executes, and `_store` drops each node's `_lazy` (`tensor.py:170-172`, `tensor.py:97`). A buffer dies when its last
    reader has run. I checked the mechanism on CPU with weak references: in a chain of four matmuls, the first
    intermediate is gone by the time the third result is stored.

    On CUDA, CuPy runs each op when called, so the held buffers were real memory: 6.85 GB above the weights before,
    0.19 GB after, for a 16 x 96 prefill. On Metal, `ctx.forward` returns lazy MLX arrays, and the GPU work happens in
    `mx.eval` after realize returns and its locals are gone. Same 3.71 GB peak either way.

??? question "How is this different from tinygrad?"
    tinygrad's scheduler groups the graph into kernels, turns reshapes and permutes into index math inside them, and
    generates and compiles Metal or CUDA source. That's a compiler. Mine groups consecutive elementwise nodes and asks MLX
    to compile each run. No codegen, no reductions or matmuls in a fused kernel, and on CUDA `fuse` is the identity
    today. It's the smallest version that tests whether fusion pays here, and the answer was 1.02x on prefill.

- [ ] I can explain `_apply`, `out_shape`, and why composite ops never read `.data`.
- [ ] I can walk `realize` step by step, including `needed` and freeing.
- [ ] I can do the 2^16 arithmetic and the prefill matmul arithmetic.

## Tokenizer (BPE)

A model reads integers, so a tokenizer cuts text into pieces from a fixed vocabulary. Byte-level BPE starts from the 256
possible bytes and learns merges: find the most frequent adjacent pair, add it as a new symbol, repeat until the
vocabulary reaches its size. The design decisions are GPT-2's: a regex pre-tokenizer so merges never cross word
boundaries, bytes as the base so no input is ever unknown, and encoding that replays merges in the order training
learned them.

Code:

- `stackgrad/tokenizer.py:10`: GPT-2's pre-tokenizer regex; `stackgrad/qwen3.py:13`: Qwen's, one digit per token.
- `stackgrad/tokenizer.py:42-83`: `train_bpe`: parallel pre-token counts, then incremental pair-count updates.
- `stackgrad/tokenizer.py:19-29`: chunk boundaries moved to the next `<|endoftext|>`.
- `stackgrad/tokenizer.py:123-131`: `_bpe`, merge the lowest-rank pair until none applies.
- `stackgrad/tokenizer.py:139-150`: `encode`, special tokens split out first.
- `stackgrad/tokenizer.py:152-162`: `encode_iterable`, constant memory.

Numbers (`bench/results/bpe.json`): a 10,000-token vocabulary trains on TinyStories in 44.7 s, 4.116 bytes per token,
and the validation set round-trips byte for byte. GPT-2's merges match tiktoken id for id, and Qwen3's match Hugging
Face's (`tests/tokenizer.py`).

??? question "Whiteboard: train BPE on low x5, lower x2, newest x6, widest x3."
    Count adjacent pairs weighted by word frequency. `e s` appears in newest (6) and widest (3), 9 total, the most, so
    merge it: newest becomes `n e w es t`. Recount: `es t` is 9, merge: `n e w est`. Next `l o` at 7 (5 from low, 2
    from lower), then `lo w` at 7. After four merges `low` is one token. Ties go to the lexicographically greater pair
    in my code (`tokenizer.py:66-68`), which is what makes training deterministic.

??? question "How many merges does a 10,000-token vocabulary need?"
    The vocabulary starts with 256 bytes plus the special tokens, here one, `<|endoftext|>` (`scripts/tinystories.py:44`).
    So 10,000 - 256 - 1 = 9,743 merges.

??? question "How did you make training fast enough?"
    Two things. Counting pre-tokens is parallel: the file is cut into chunks whose boundaries move to the next
    `<|endoftext|>`, and a process pool counts each (`tokenizer.py:44-50`). Then merging works on distinct pre-tokens
    with their counts, not the corpus, and updates pair counts in place: a `where` index maps each pair to the words
    containing it, so a merge touches only those words (`tokenizer.py:55-82`). The remaining shortcut is the
    `max` over all pairs per merge, marked in the code; a lazy max-heap would replace it if training time mattered.

??? question "Encoding merges the lowest-rank pair first. Why not the most frequent pair in the input?"
    Rank is the order training learned the merges. Replaying that order reproduces how training would have tokenized
    the same text, and it doesn't depend on the input, so the same text always gives the same ids. Frequency in the input
    would make a word's tokens depend on what else is in the document.

??? question "Why pre-tokenize with a regex at all?"
    It stops merges from crossing word and punctuation boundaries, so you don't get tokens like "dog." glued to the next
    word's start. It also makes encoding cacheable: the tokenizer memoizes each pre-token's ids (`tokenizer.py:133-137`),
    and natural text repeats the same pre-tokens over and over.

??? question "Why does Qwen split every digit into its own token?"
    Its pre-tokenizer matches `\p{N}` one character at a time (`qwen3.py:13`). Numbers then tokenize the same way
    everywhere: 1234 is always 1, 2, 3, 4. With multi-digit tokens, 1234 might be "12" "34" in one place and "123" "4"
    in another, which makes arithmetic harder to learn.

??? question "How do special tokens work?"
    A regex of the special strings, longest first, splits the text before the pre-tokenizer runs
    (`tokenizer.py:13-16`). Odd slots of the split are special tokens and map straight to their ids; even slots go
    through BPE (`tokenizer.py:143-149`). Ordinary text can never produce them, since BPE never sees the special strings
    as a unit.

??? question "How does encode_iterable stream in constant memory?"
    It holds back the last pre-token of each chunk, because more text could re-split it (a word cut in half by a chunk
    boundary), encodes everything before it, and carries the tail into the next chunk (`tokenizer.py:152-162`).

- [ ] I can run BPE training by hand on the four-word example.
- [ ] I can explain rank-order encoding and why it's deterministic.

## Transformer

One token's vector is the model's notes on that token, and each layer reads the notes and adds a correction to the
residual stream. Attention mixes information across positions; the feed-forward works on each position alone. The
design decisions are the modern ones CS336 teaches: RMSNorm before each sublayer (pre-norm), RoPE for position, SwiGLU
for the feed-forward, and grouped-query attention. One class, `TransformerLM`, covers the story model and Qwen3; Qwen3
adds per-head RMSNorm on queries and keys, a head width (128) that differs from `d_model / n_heads` (64), and tied
input and output embeddings.

Code:

- `stackgrad/gpt.py:20-39`: `Block`, pre-norm, plus the post-norm and no-norm ablations.
- `stackgrad/layers.py:170-178`: `RMSNorm`; `stackgrad/layers.py:205-214`: `SwiGLU`.
- `stackgrad/attention.py:9-21`: `rope_tables` and `apply_rope`, half-split layout.
- `stackgrad/attention.py:77-102`: `Attention.forward`, GQA, qk-norm, cache, padding.
- `stackgrad/mlops.py:301-352`: the `Attention` primitive, causal mask and composed fallback.
- `stackgrad/attention.py:24-30`: padding mask with -1e9.

Numbers: story model 22,696,448 parameters, Qwen3-0.6B 596,049,920 (post table, `bench/results/qwen3_parity.json`).
Ablations at 10,000 steps, baseline 1.5242: post-norm 1.5373, SiLU 1.5538, NoPE 1.6089, no RMSNorm diverged at step 627,
no RMSNorm at lr 3e-4 1.6994 (`bench/results/ablations.json`).

??? question "Walk one token through a block."
    `x = x + attn(rmsnorm(x))`, then `x = x + ffn(rmsnorm(x))` (`gpt.py:38-39`). Inside attention: project to q, k, v,
    reshape into heads, apply qk-norm for Qwen3, rotate q and k by their positions with RoPE, write k and v into the
    cache if there is one, then causal attention and the output projection (`attention.py:87-102`). The feed-forward is
    `w2(silu(w1 x) * w3 x)`. Nothing ever overwrites the residual stream; each sublayer adds to it.

??? question "RMSNorm vs LayerNorm?"
    LayerNorm subtracts the mean, divides by the standard deviation, then scales and shifts. RMSNorm skips the mean and
    the bias: `x * (mean(x^2) + eps)^-0.5 * g` (`layers.py:178`). That drops one reduction and one parameter vector, and
    Llama and Qwen use it. Its job in a pre-norm block is to hand each sublayer inputs of a standard size, whatever the
    residual stream's scale has grown to.

??? question "Whiteboard: why does RoPE make a score depend only on the distance between positions?"
    Split q and k into pairs and treat each pair as a 2-D vector. Position p rotates each pair by angle `p * theta_i`.
    A rotation matrix satisfies `R(a)^T R(b) = R(b - a)`, so the dot product of a query at position m and a key at
    position n is `q^T R(m)^T R(n) k = q^T R(n - m) k`. Only `n - m` survives. Each pair rotates at its own speed,
    `theta^(-i/half)`, so fast pairs separate neighbors and slow pairs track long distances (`attention.py:9-14`).

??? question "Why the half-split layout, and what breaks if you get it wrong?"
    Llama and Qwen pair dimension i with i + D/2, not with i + 1 (`attention.py:17-21`). The weights were trained with
    that pairing, so the interleaved layout would rotate the wrong numbers together. Nothing crashes. The logits come out
    wrong, which is what the Hugging Face parity test catches.

??? question "What does GQA save, and how does the backward handle shared heads?"
    Qwen3-0.6B has 16 query heads and 8 key-value heads, so the KV cache is half what full multi-head attention would
    store. In the composed path, keys and values are repeated across each group of query heads
    (`mlops.py:306`). In the backward, each KV head sums the gradients of the query heads it served:
    `dk.reshape(B, Hkv, rep, S, D).sum(axis=2)` (`mlops.py:345-346`).

??? question "Whiteboard: count the story model's parameters."
    Vocabulary 10,000, width 512, 4 layers, feed-forward 1,344, untied head. Embedding 10,000 x 512 = 5,120,000, head the
    same. Per layer: attention 4 x 512 x 512 = 1,048,576; SwiGLU 3 x 512 x 1,344 = 2,064,384; two RMSNorm gains 1,024;
    total 3,113,984. Four layers 12,455,936, plus the final norm's 512. Sum: 22,696,448. I checked it by building the
    model on CPU.

??? question "Why mask padding with -1e9 instead of -inf?"
    A left-padding query sees only padding keys. If every score in its row is -inf, softmax computes 0/0 = NaN, and
    later `0 * NaN = NaN` leaks into real rows (`attention.py:28`). -1e9 gives a uniform softmax over the padding instead,
    finite and ignored. The causal mask can use -inf because every query sees at least its own key.

??? question "Why scale scores by 1/sqrt(d)?"
    A dot product of two random d-dimensional vectors with unit-variance entries has variance d. Without the scale,
    scores grow with head width, softmax saturates toward one-hot, and its gradients vanish. Dividing by sqrt(d) keeps
    the score variance near 1.

??? question "What did the ablations show?"
    At 10,000 steps against a 1.524 baseline: post-norm adds 0.013, a plain SiLU feed-forward of the same size adds 0.03,
    and no position information adds 0.085. NoPE still learns because the causal mask leaks position: each token sees a
    different number of earlier tokens, and a head can count them. No RMSNorm at lr 1e-3 diverged at step 627; at 3e-4
    it trained to 1.699.

??? question "How did you make the SiLU ablation a fair comparison?"
    SwiGLU has three matrices, a plain MLP two. At `d_ff = 2048` the SiLU MLP has 2 x 512 x 2,048 = 2,097,152 weights
    per layer against SwiGLU's 3 x 512 x 1,344 = 2,064,384, so the comparison is about the gate and not parameter count
    (`train_gpt.py:51`).

- [ ] I can draw a pre-norm block and write its two lines.
- [ ] I can prove RoPE's relative-position property.
- [ ] I can count the story model's parameters on a whiteboard.

## Training

Training repeats one step 40,000 times: sample 32 random windows of 256 tokens, compute cross-entropy, backward, clip,
AdamW. The decisions are CS336's: AdamW with betas (0.9, 0.95) and weight decay 0.1, linear warmup over 1,000 steps then
cosine decay from 1e-3 to 1e-4, and gradient clipping at norm 1.0.

Code:

- `train_gpt.py:91-102`: the loop; `train_gpt.py:24-29`: `get_batch`.
- `stackgrad/optim.py:43-79`: Adam and AdamW with decoupled decay.
- `stackgrad/optim.py:82-89`: `cosine_lr`; `stackgrad/optim.py:92-101`: `clip_grad_norm`.
- `train_gpt.py:82-83, 107-110`: the 6N FLOP count and MFU.
- `stackgrad/loss.py:20-28`: cross-entropy through log-softmax.

Numbers:

- First step loss 9.2335 against ln(10,000) = 9.21; validation 2.807 at step 500, 1.399 at step 40,000; 4,257.7 s
  total; median 8.50 TFLOP/s, MFU 23.9% against a 35.6 TFLOP/s rated peak (`runs/tinystories/metrics.jsonl`,
  `runs/tinystories/config.json`).
- Scaling laws: compute-optimal sizes 3.0M, 5.5M, 8.2M parameters at 3e14, 1e15, 3e15 FLOPs
  (`bench/results/scaling.json`); at the full run's compute the fit asks for 24M parameters on 237M tokens (post).

??? question "Why is the first loss 9.23?"
    An untrained model spreads probability about equally over the vocabulary, and cross-entropy of a uniform guess over 10,000
    tokens is `ln(10,000) = 9.21`. Step 1 logged 9.2335. If your first loss is far from `ln(V)`, your init or your loss is
    wrong.

??? question "AdamW vs Adam with L2 regularization?"
    L2 adds `wd * w` to the gradient, and Adam then divides that by `sqrt(v)`, so weights with large gradient variance
    get less decay. AdamW shrinks the weight itself, `w *= 1 - lr * wd`, outside the adaptive step
    (`optim.py:75-79`). Decay then means the same thing for every weight.

??? question "Why warm up the learning rate?"
    Early on, Adam's moment estimates rest on a handful of steps and the weights are random, so the gradient direction is
    noisy. A full-size step then can push the model somewhere bad. Ramping over 1,000 steps lets the averages settle
    first.

??? question "How does clip_grad_norm work, and what does it cost?"
    It computes the global norm over all parameters' gradients and, if it exceeds the max, scales every gradient by
    `max / norm` (`optim.py:92-101`). It's one device sum and one host sync for the whole model, not one per parameter.

??? question "Whiteboard: compute the MFU of the TinyStories run."
    Training FLOPs per token are about 6N: 2N forward, 4N backward. N counts matmul weights only, since the embedding is a
    lookup: 22,696,448 - 5,120,000 = 17,576,448. The last step ran 80,551 tokens/s, so
    6 x 17,576,448 x 80,551 = 8.49 TFLOP/s. The RTX 3090's rated TF32 peak is 35.6, so MFU is 23.9%. 6N also ignores
    attention's score FLOPs, which at T = 256 are small next to the weights.

??? question "Where does the other 76% go?"
    The profile of one CUDA step shows it (`bench/results/profile_cuda.json`). The profiler syncs after every op, so the
    step takes 76.9 ms profiled against 69.0 ms unprofiled. Matmuls and their backward take 32.6 ms of it. Elementwise
    ops (Mul, Add, Sigmoid, Expand, Reshape and their backwards) are memory-bound and add up, and 7.6 ms sits between
    ops, which is Python and launch overhead. Each of those small ops pays a fixed launch cost whatever its size.

??? question "Scaling laws: was the story model the right size?"
    I ran IsoFLOP curves: three budgets from 3e14 to 3e15 FLOPs, five sizes each, and fit the best size per budget. The
    optima came out at 3.0M, 5.5M, and 8.2M parameters. Extrapolated to the full run's compute, the fit asks for 24M
    parameters on 237M tokens; the story model has 17.6M matmul weights and saw 328M tokens. A little small for its
    compute.

??? question "What's the shortcut in get_batch?"
    Window starts are drawn from `[0, len - context - 1)`, so the data's last token is never a target
    (`train_gpt.py:25-26`). Fixing it changes the random batches behind every published run, so it stays, marked.

??? question "What isn't in your checkpoints?"
    Weights only. No AdamW moments, no step counter, no `--resume` (cut 2026-11-16 in SPEC; no run in the post resumes).
    A crash mid-run means restarting. In a real training job I'd save optimizer state and the data position too.

- [ ] I can explain AdamW, warmup, cosine decay, and clipping in one sentence each.
- [ ] I can do the MFU arithmetic from tokens/s.
- [ ] I can explain the step-627 divergence (see the debugging stories).

## GPUs and backends

Each op in `mlops.py` receives a module `xp` with the same small set of functions, NumPy's names, and one copy of every
gradient runs on all three backends. The decision is to keep backends thin: CPU is NumPy, Metal is MLX re-exports plus
the places MLX differs (immutable arrays, laziness), CUDA is CuPy plus Triton kernels.

Code:

- `stackgrad/backend/__init__.py:7-11`: `backend.get(device)` imports the module.
- `stackgrad/backend/metal.py:82-91`: `realize = mx.eval`, cache cap, `fuse`, `capture`.
- `stackgrad/backend/metal.py:94-95`: `scatter_add` with `.at[].add`, since MLX arrays are immutable.
- `stackgrad/optim.py:19-21`: the optimizer realizes weights and state each step.
- `stackgrad/module.py:62-64`: `to()` realizes, because a lazy MLX cast belongs to its thread's stream.
- `stackgrad/tensor.py:29-41` and `stackgrad/layers.py:33-36`: autocast; `stackgrad/optim.py:104-121`: `LossScaler`.
- `stackgrad/sampling.py:29-39`: sampling on device with first-crossing.

Numbers:

- One elementwise multiply of 1,000 numbers: 3.09 us NumPy, 7.59 us Metal, 41.89 us CUDA; of 10M: 863, 778, 99.7 us
  (`bench/results/backend_overhead_cpu_metal.json`, `bench/results/backend_overhead_cuda.json`).
- Matmul peaks: 33.9 TFLOP/s TF32 on the 3090, 3.62 TFLOP/s bfloat16 on the M4 (same files).
- Training step: CPU 2,282.9 ms, CUDA 90.9 ms (25.1x); Metal 676.0 ms vs plain MLX 554.8 ms (82%) (same files).
- Mixed precision step: CUDA float32 69.6 ms vs float16 61.8 ms; Metal float32 668.7 vs bfloat16 663.5
  (`bench/results/mixed_precision_cuda.json`, `bench/results/mixed_precision_metal.json`).

??? question "Why can a GPU be slower than NumPy?"
    Every kernel launch pays a fixed cost while the CPU describes the work and the GPU schedules it. For 1,000 numbers
    that fixed cost is the whole cost: 42 us on the 3090 against 3.1 us in NumPy. At 10M numbers the work buries the
    launch, and the 3090 wins 8.7x (863 vs 99.7 us). A GPU pays off when each kernel gets enough work.

??? question "Whiteboard: arithmetic intensity of a matmul vs an elementwise op."
    An n x n matmul in float32 reads and writes 3n^2 numbers, 12n^2 bytes, and does 2n^3 FLOPs: n/6 FLOPs per byte. At
    n = 4,096 that's 683. An elementwise add does 1 FLOP per 12 bytes. The M4's ridge point, measured peak over measured
    bandwidth, is 3.62 TFLOP/s / 101.4 GB/s = 36 FLOPs per byte (arithmetic from `bench/results/backend_overhead_cpu_metal.json`
    and `bench/results/quant.json`). Above it you're compute-bound, below it memory-bound. Big matmuls sit far above;
    elementwise ops and decode sit far below.

??? question "What's different about MLX?"
    Two things. Arrays are immutable, so scatter-add uses `.at[key].add` and the KV cache write uses `slice_update`
    (`metal.py:94-95`, `metal.py:107-109`). And MLX is lazy too: ops build a graph that runs on `mx.eval`. So the
    optimizer calls `realize` on the weights every step, or the graph would grow across steps (`optim.py:19-21`).
    Activation checkpointing saved nothing on Metal until it forced evaluation at each block boundary, because the lazy
    graph kept every activation alive anyway (`checkpoint.py:12-13`).

??? question "Why does Module.to() force evaluation?"
    A lazy MLX cast is tied to the stream of the thread that recorded it. The serving engine runs on its own thread, so
    a pending weight cast from the loading thread would run on the wrong stream there. `to()` realizes every moved
    tensor before returning (`module.py:62-64`).

??? question "What does autocast do, and why does float16 need loss scaling?"
    Inside `autocast`, every Linear casts its input and weight to the low-precision type, multiplies, and casts back to
    float32 (`layers.py:33-36`). Everything else stays float32. float16 has a narrow exponent range, so small gradients
    underflow to zero. `LossScaler` multiplies the loss by 2^15, divides the gradients back, halves the scale and skips
    the step on overflow, and doubles it after 1,000 clean steps (`optim.py:104-121`). bfloat16 has float32's exponent
    range and needs none. CuPy has no bfloat16, so the CUDA path uses float16.

??? question "Why does stackgrad reach 82% of plain MLX and not 100%?"
    The plain MLX model uses MLX's own `nn.RMSNorm`, `nn.RoPE`, and fused attention (`bench/backend_overhead.py:126-150`),
    where I compose RMSNorm, RoPE, and the loss from primitives. And every stackgrad op adds a Python record in `_apply`
    and a Python step in `realize` on top of MLX's own dispatch. Same model, 676.0 vs 554.8 ms per step. The biggest
    single item I can point to is attention's backward, which is composed on Metal and took 120.9 ms of a 776.7 ms
    profiled step (`bench/results/profile_metal.json`).

??? question "Why sample on the device, and what's first-crossing?"
    At 64 rollout rows, copying logits to the host and sampling there took 98.5 ms per step against 1.2 ms on device,
    more than the 84.3 ms model step itself (`bench/results/rollout_cuda.json`). On device you compute a CDF with a
    parallel cumsum. In float32 a parallel cumsum isn't guaranteed monotonic, so "count entries below u" picks wrong
    tokens: 714 of 20,480 draws disagreed with exact sampling. Taking the first index where the CDF crosses u disagreed
    in 2 (`bench/results/sampling_cuda.json`, `sampling.py:38-39`).

??? question "Attention backward takes 14.5x its forward on Metal. Why so lopsided?"
    The profile shows 120.9 ms backward vs 8.3 ms forward (`bench/results/profile_metal.json`). The forward is my fused
    v3 kernel. Metal has no fused backward, so after a fused forward the backward rebuilds P from the saved
    log-sum-exp by recomputing the full score matrix, repeats K and V across the query groups, and runs composed
    matmuls (`mlops.py:333-346`). That writes and reads the score-sized arrays the forward avoided. On CUDA the Triton
    backward keeps it at 2.95 ms vs 0.73 ms (`bench/results/profile_cuda.json`).

- [ ] I can explain launch overhead with the 1,000 vs 10M numbers.
- [ ] I can compute arithmetic intensity and the ridge point.
- [ ] I can list how the MLX backend differs from NumPy and why.

## Attention kernels

Composed attention writes the full score matrix to memory and reads it back in the next kernel. FlashAttention brings
tiles of Q, K, and V on chip, finishes attention for each tile there, and never writes scores out. The obstacle is the
softmax, which seems to need the whole row's max and sum. The one-pass softmax keeps a running max m and a running sum
l counted relative to m, and rescales l (and the output accumulator) by `exp(m_old - m_new)` when a bigger score
arrives.

stackgrad has two Metal kernels and one Triton kernel family:

- v3 (prefill and training): a threadgroup of 4 simdgroups x 8 query rows = 32 rows. Scores and P @ V use the GPU's
  8 x 8 `simdgroup_multiply_accumulate`; four lanes share each softmax row.
- v1 (decode, rewritten 2026-11-21): a threadgroup per query row, D lanes. Lane t scores key `j0 + t` of each tile in
  parallel, then every lane runs the tile's softmax in key order and keeps output dimension t.
- v2 (deleted): 32 threads, one query row each, sharing K/V tiles in threadgroup memory. v3 beat it at every shape.
- Triton FlashAttention-2 forward plus two backward kernels, on the CuPy path without torch.

Code:

- `stackgrad/backend/metal_attention.py:10-33`: v1; `stackgrad/backend/metal_attention.py:36-131`: v3;
  `stackgrad/backend/metal_attention.py:140-156`: launch, `BK = 32 if D <= 64 else 16`.
- `stackgrad/backend/metal.py:112-118`: v1 when `T == 1`, else v3, `D % 8 == 0` and `D <= 128`.
- `stackgrad/backend/cuda_attention.py:21-45`: forward; `:48-74`: dK/dV; `:77-97`: dQ; `:122-138`: launch and GQA sum.
- `stackgrad/mlops.py:330-346`: the composed backward, the reference for both.

Numbers (`bench/results/flash_attention.json`, M4, float32 unless noted):

| Shape | materialized | old v1 | v2 | v3 | mx.fast.sdpa |
|---|---|---|---|---|---|
| CS336 training (B32 H16 T256 D32) | 9.94 ms | 3.02 | 2.91 | 1.49 | 9.94 |
| Qwen3 prefill 2k (D128) | 26.3 | 329 | 124 | 18.8 | 7.19 |
| decode, 2k cache, bfloat16 | 6.91 | 1.02 | 33.9 | 6.01 | 0.712 |

The "old v1" column is the deleted thread-per-row design; the results file has no row for the rewritten v1 yet. Triton
at the training shape: 0.764 ms forward plus backward against 6.477 ms composed, 8.5x
(`bench/results/flash_attention_cuda.json`). All Metal kernel runs within 1e-4 of a float64 reference, 10 of 10
(`bench/results/backends_metal.json`).

??? question "Whiteboard: run the online softmax on scores 1, 3, 2."
    Start m = -inf, l = 0. Score 1: m = 1, l = e^0 = 1. Score 3: new max, so convert the old sum at the rate
    e^(1-3) = 0.135: l = 1 x 0.135 + e^0 = 1.135, m = 3. Score 2: no new max, l = 1.135 + e^(2-3) = 1.503. Direct:
    e^-2 + e^0 + e^-1 = 1.503. Same.

    The output works the same way: keep `acc = sum p_j v_j` with `p_j = e^(s_j - m)`, multiply acc by the same
    correction whenever m changes, add the new `p v`, and divide by l at the end. In my v1 that's three lines:
    `l = l * corr + p; acc = acc * corr + p * v[j]; m = m_new` (`metal_attention.py:25-28`).

??? question "What does FlashAttention save at the story model's shape?"
    Arithmetic: batch 32, 16 heads, 256 x 256 scores per head is 33.5M scores, 134 MB in float32, written by one kernel
    and read back by the next, then the same again for the probabilities. The fused kernel keeps a tile of scores in
    threadgroup memory and three running values per row. Measured: 1.49 ms vs 9.94 ms for materialized scores.

??? question "Walk through v3's structure."
    One threadgroup handles 32 query rows of one (batch, head). It stages Q, scaled, through threadgroup memory into
    8 x 8 matrix registers, four simdgroups of 8 rows each. For each key tile of BK keys it loads K and V into threadgroup
    memory, computes the 8 x BK score block per simdgroup with `simdgroup_multiply_accumulate`, applies the causal mask,
    and runs the online softmax: lanes 4r to 4r+3 own row r, a quarter of the columns each, and reduce the max and sum
    with `simd_shuffle_xor`. It rescales O by a diagonal matrix of corrections and accumulates P @ V with the matrix
    units again. At the end it divides by l and writes the log-sum-exp for the backward (`metal_attention.py:36-131`).

??? question "Why did you delete v2?"
    v3 beat it at every shape in the table, and the 3,000-line budget makes every kernel pay rent. v2 also had the worst
    decode number: 33.9 ms on the 2k-cache bfloat16 shape, because a one-row query leaves 31 of its 32 threads idle. The
    agent that built the fixed-capacity cache deleted it to pay for its lines, which I reviewed.

??? question "Why a separate decode kernel, and how does the rewritten v1 work?"
    Decode has one query row per sequence and head. v3's 32-row threadgroups would leave 31 of 32 rows idle. The old v1 used one
    thread per query row, so a one-token step at batch 1 ran 16 threads for Qwen3's 16 heads, and after the cache change
    that kernel took about half of a 20 ms step (session log; the post states the 16-thread problem).

    The new v1 launches a threadgroup of D threads per query row (`metal_attention.py:145-146`). Each tile covers D keys:
    lane t computes the dot product of q with key `j0 + t`, writes it to threadgroup memory, then after a barrier every
    lane walks the tile's D scores in key order, updating m, l, and its own output dimension t
    (`metal_attention.py:18-30`). Every lane computes the same m and l in the same order a single thread would, so the
    sums round the same way as the old kernel. The agent reported bitwise equality with the old v1 on 14 shape and dtype
    cases.

??? question "Why does MLX beat v3 by 2.6x at Qwen3's 128-wide heads, when v3 beats MLX 6.7x on the story model?"
    MLX has a dedicated single-kernel attention for 128-wide heads but none for 32-wide, where it falls back to
    something as slow as materializing scores (9.94 ms in both columns). At D = 128 my v3 also shrinks its key tile to
    BK = 16 to keep K and V tiles within 32 KB of threadgroup memory (`metal_attention.py:144`), so it gets half the
    keys per tile pass.

??? question "FlashAttention-2 backward: what do you save, and what do you recompute?"
    The forward saves O and one number per row, the log-sum-exp `lse = m + log(l)`. The backward first computes
    `delta = rowsum(dO * O)` per row (`cuda_attention.py:127`). Each tile then rebuilds `P = exp(S - lse)` on chip,
    computes `dV += P^T dO`, `dP = dO V^T`, `dS = P * (dP - delta)`, and `dK += dS^T Q * scale`, `dQ += dS K * scale`.
    Recomputing S costs one extra matmul per tile, cheaper than storing and reading a score-sized P.

??? question "Whiteboard: derive the softmax backward and the delta trick."
    For one row, `p = softmax(s)`, upstream gradient dp. The Jacobian is `dp_i/ds_j = p_i (1[i=j] - p_j)`, so
    `ds_j = sum_i dp_i p_i (1[i=j] - p_j) = p_j (dp_j - sum_i p_i dp_i)`. That's `ds = p * (dp - rowsum(p * dp))`. With
    `dp = dO V^T` and `O = P V`, `rowsum(p * dp) = sum_j p_j (dO . v_j) = dO . (sum_j p_j v_j) = dO . O`. So the row term
    needs only dO and O, no P (`mlops.py:341-342`).

??? question "Why two backward kernels instead of one?"
    dK and dV for a key block sum over every query block that sees it; dQ for a query block sums over key blocks. One
    kernel would need atomic adds for one of them. Two kernels each own their output: `_backward_kv` loops over query
    blocks per key block, `_backward_q` loops over key blocks per query block (`cuda_attention.py:48-97`). The kv kernel
    also skips query blocks that end before its first key (`cuda_attention.py:60`).

??? question "How do the kernels handle GQA and the 3090's shared memory?"
    Query head h reads KV head `h // (H / Hkv)` (`cuda_attention.py:24`). The backward writes dK and dV per query head
    and the caller sums each group into its KV head (`cuda_attention.py:137-138`). For shared memory, D = 128 at 64 x 32
    blocks needs 114,688 bytes and the 3090 has 101,376, so D = 128 uses 32 x 32 blocks (`cuda_attention.py:100-102`).
    The backward runs one pipeline stage because three stages overflow shared memory (`cuda_attention.py:134`).

??? question "How do you launch Triton on CuPy arrays without importing torch?"
    Triton reads two things from a tensor argument: `data_ptr()` and `dtype`. A tiny `_Arg` wrapper supplies them from a
    CuPy array (`cuda_attention.py:12-18`).

- [ ] I can run the online softmax recurrence by hand, output accumulator included.
- [ ] I can derive the softmax backward and the delta identity.
- [ ] I can explain v1, v3, why v2 died, and the FA2 backward split.

## Loading Qwen3

A strong test of a framework is someone else's model. stackgrad reads Qwen3-0.6B's safetensors, maps Hugging Face's
tensor names onto its own parameters, and refuses to run on any mismatch. The decisions: memory-map the file and convert
one tensor at a time, widen bfloat16 to float32 by shifting bits (lossless), assert that every parameter gets a
checkpoint tensor, and since 2026-11-21 skip the random initialization the checkpoint overwrites (SKIP_INIT).

Code:

- `stackgrad/state.py:12-27`: safetensors: 8-byte header length, JSON header, `np.memmap`, bf16 shift.
- `stackgrad/qwen3.py:16-22`: config mapping; `stackgrad/qwen3.py:25-38`: names and transposes.
- `stackgrad/qwen3.py:41-69`: `load_qwen3`, with SKIP_INIT at 44-48 and the asserts at 52, 62, 64, 66, 68.
- `stackgrad/layers.py:10-14`: `SKIP_INIT` and `normal()`; `stackgrad/gpt.py:63-64`: skip the CS336 init.
- `stackgrad/qwen3.py:72-80`: tokenizer with special tokens at their own ids; `:90-108`: chat template.

Numbers (`bench/results/qwen3_parity.json`): 17-token prompt, largest logit 24.29. Largest gap: CPU float32 5.05e-5,
Metal float32 9.73e-5, Metal bfloat16 0.346 (1.4% of the largest logit), next-token agreement 100% in all three, greedy
decoding matches Hugging Face for 24 tokens. The random init was 14.7 s of a 19.6 s load before SKIP_INIT (comment at
`stackgrad/qwen3.py:44`; no result file).

??? question "How does your safetensors reader work?"
    The first 8 bytes are a little-endian header length; the header is JSON mapping each tensor name to a dtype, shape,
    and byte range. I `memmap` the rest of the file, so nothing loads until touched, and yield one tensor at a time
    (`state.py:12-27`). Loading one tensor at a time matters: Qwen3-1.7B's float32 weights beside a full copy would need
    16 GB (`qwen3.py:55`).

??? question "Whiteboard: why is bfloat16 to float32 lossless?"
    float32 is 1 sign bit, 8 exponent bits, 23 mantissa bits. bfloat16 is the top 16 of those: sign, the same 8 exponent
    bits, 7 mantissa bits. So shifting the 16 bits left by 16 and appending zeros rebuilds an exact float32
    (`state.py:22-24`). Same range as float32, two to three significant digits, half the memory.

??? question "Whiteboard: how many bytes are Qwen3-0.6B's weights in bfloat16?"
    596,049,920 parameters x 2 bytes = 1.19 GB, which matches the 1,192 MB weight read per decode token in
    `bench/results/quant.json`.

??? question "Why transpose some tensors, and what if you forget one?"
    Hugging Face stores Linear weights as (out, in); stackgrad stores (in, out) so the forward is `x @ W`
    (`qwen3.py:26`). For q_proj, (2,048, 1,024) vs (1,024, 2,048), the shape assert catches a missing transpose. For
    k_proj and v_proj, 1,024 to 1,024, the matrix is square, the shape check passes, and the numbers come out wrong
    with no error. That's why the parity test against Hugging Face's logits exists.

??? question "bfloat16 shows a 0.35 gap but the same tokens. Is that a bug?"
    No. bfloat16 keeps about 2 to 3 significant digits, and errors compound over 28 layers. 0.35 on a largest logit of
    24.3 is 1.4%. The ranking at the top survives, so the next token agrees at every position. In float32 the gap is
    5.1e-5, rounding error. The float32 number tests correctness; the bfloat16 number tests the format people run.

??? question "What is SKIP_INIT, and what stops it from shipping garbage weights?"
    Building `TransformerLM` used to draw random weights for 596M parameters, then the loader overwrote every one. With
    `SKIP_INIT.on`, `normal()` returns `np.empty` (uninitialized memory) and `TransformerLM` skips its init loop
    (`layers.py:13-14`, `gpt.py:63-64`). The guard is the loader's asserts: the model's parameter names must equal the
    mapping's names, every checkpoint tensor must map to one, and nothing may stay unplaced (`qwen3.py:52, 62, 66`). A
    parameter the checkpoint lacked would fail the load instead of keeping garbage. The flag is thread-local and reset in
    a `finally`, so a model built on another server thread meanwhile still gets a real init.

??? question "What does Qwen3 have that the story model doesn't?"
    Per-head RMSNorm on q and k, head width 128 where `d_model / n_heads` would be 64, 8 KV heads for 16 query heads,
    rope theta 1e6, RMSNorm eps 1e-6, and tied embeddings, where the output head is the embedding table transposed
    (`qwen3.py:16-22`, `gpt.py:122-124`). The loader also checks that a checkpoint's stored `lm_head` equals the
    embedding (`qwen3.py:67-68`).

??? question "How do you know the chat template is right?"
    `chat_prompt` reproduces Hugging Face's template, including how tool definitions and tool results render
    (`qwen3.py:90-108`), and `tests/agent.py` compares against it. The `<think>` block is closed and empty when thinking is
    off, as Qwen's template does.

- [ ] I can describe the safetensors layout and the bf16 bit trick.
- [ ] I can explain why a square matrix makes the transpose bug invisible to shape checks.
- [ ] I can explain SKIP_INIT and the asserts that make it safe.

## Serving

Generating a reply takes one forward pass per new token. The KV cache stores each earlier position's keys and values
so a step computes only the new token's. Serving many people means managing that memory and keeping the GPU busy. The
decisions: take vLLM's block bookkeeping but store each cache as one array (bookkeeping without paging), continuous
batching that admits and retires requests between steps, preemption of the newest request when blocks run out, and one
batched decode pass over left-padded caches.

Code:

- `stackgrad/serve/kv_cache.py:1-35`: `BlockManager`: free list, block tables, lengths.
- `stackgrad/serve/scheduler.py:39-69`: `step`: admit, prefill, preempt, decode, retire.
- `stackgrad/serve/runner.py:16-29`: prefill one sequence, decode the batch; `:36-55`: join and split padded caches.
- `stackgrad/serve/engine.py:24-53`: background thread, per-request queues, error fan-out.
- `stackgrad/serve/api.py:189-201`: the server: Qwen3 in bfloat16 on Metal, `BlockManager(512, 16)`.
- `stackgrad/backend/metal.py:84`: the 1 GB MLX buffer-cache cap.

Numbers (`bench/results/serve_batch_metal.json`, Qwen3-0.6B bfloat16, 128 new tokens each):

| Chats | one pass per chat | one batched pass |
|---|---|---|
| 1 | 63.9 tok/s | 64.0 tok/s (15.09 ms/step) |
| 2 | 64.8 | 52.4 |
| 4 | 64.2 | 96.3 |
| 8 | 63.4 | 154.9 (47.87 ms/step) |

Batched replies match solo ones 8 of 8 in float32 and 4 of 8 in bfloat16 (same file). Scheduler scenarios, including a
tiny KV budget with 3 preemptions, give every request its solo tokens (`bench/results/serve_metal.json`).

??? question "Whiteboard: KV cache bytes per token for Qwen3-0.6B."
    Keys and values, times 28 layers, times 8 KV heads, times 128 numbers per head, times 2 bytes in bfloat16:
    2 x 28 x 8 x 128 x 2 = 114,688 bytes, 112 KiB per token. A 4,096-token context is 470 MB per sequence. The server's
    pool of 512 blocks of 16 tokens holds 8,192 tokens, 940 MB when full. GQA is why it's this small: 16 KV heads would
    double it.

??? question "What does paged attention buy, and what did you take from it?"
    Reserving the maximum context per request wastes memory on tokens never written. Paging hands out fixed-size blocks
    as a sequence grows, so memory tracks actual length. I took the bookkeeping: `BlockManager` hands out 16-token
    blocks from a free list, `blocks_needed` is a ceiling division, and release returns them (`kv_cache.py:10-32`). The
    scheduler uses it for admission and preemption. The caches themselves are contiguous arrays, so I skipped the
    paged attention kernel that reads K and V through block tables. The cost shows in the runner: a change in batch
    membership copies every live cache to re-pad it (`runner.py:23`).

??? question "Continuous vs static batching?"
    Static batching runs a batch until its longest reply finishes, so finished slots sit idle. Continuous batching
    re-forms the batch every step: new requests join between steps, finished ones leave right away
    (`scheduler.py:42-51`, `:71-75`). Decode is cheap to batch, because a step reads every weight once whether it serves
    one chat or eight.

??? question "Walk through one scheduler step."
    Admit from the waiting queue while the batch has fewer than 8 requests and the block manager can hold the request's
    tokens. A preempted request replays its prompt plus what it already generated. Each admitted request is prefilled
    and gets its first token from the prefill logits. Then for every running request that needs a decode, make room for
    one more token; if the blocks run out, preempt the newest running request until it fits. Run one batched decode,
    sample, and retire finished requests, releasing their blocks (`scheduler.py:39-69`).

??? question "Why preempt the newest request, and what does preemption cost?"
    The newest has the least work invested, so recomputing it wastes the least, and the oldest requests keep making
    progress, which avoids starvation. The cost is a re-prefill of prompt plus generated tokens when it comes back. It
    goes to the front of the waiting queue (`scheduler.py:83-88`).

??? question "Why is batching nearly free in decode?"
    A decode step is bound by reading the weights, about 1.2 GB per step for this model. Eight rows read the same weights
    once. Measured: 47.87 ms for a step of 8 tokens against 15.09 ms for 1, so 8x the tokens for 3.2x the time.

??? question "Why is 2-chat batching slower than two separate passes?"
    Two chats of different lengths need left padding, and padding pushes the step off both fast paths. Capture requires
    `pad is None` (`gpt.py:77`), and the fused attention kernels take no mask, so a padded batch runs composed attention
    (`mlops.py:317`). One chat runs captured at 15.09 ms per step; two padded chats run eager. At 4 and 8 chats the
    batching win outgrows that penalty. A paged decode kernel that takes per-row lengths would let padded batches run
    captured.

??? question "Why do batched bfloat16 replies differ from solo replies, when float32 matches?"
    Padding changes the shapes the reductions run over, so the rounding differs. In bfloat16, with 2 to 3 significant
    digits, that's enough to flip a greedy argmax now and then, and once one token differs the replies diverge. In
    float32 all 8 replies match their solo runs. The scheduler tests use exact comparison and pass on both CPU and Metal.

??? question "What happens when a step throws?"
    The engine catches it, releases every request's blocks and caches, and puts the exception into every waiting
    caller's queue, including requests submitted during the failed step, then clears the queues
    (`engine.py:34-46`). Without that, a caller would wait forever on a queue nobody writes to.

??? question "Why cap MLX's buffer cache at 1 GB?"
    MLX keeps freed buffers to reuse them. While the KV cache still grew every step, the freed buffers came in new sizes
    the cache couldn't reuse, and 4-chat serving fell to 21.9 tok/s. Capping the cache brought it to 91.9 (2026-11-21
    session log). With today's fixed-capacity cache, 8-chat serving holds 2.4 GB cached uncapped vs 1.1 GB capped, at the
    same speed (`metal.py:84`). The full story is in the debugging section.

- [ ] I can compute KV bytes per token and the pool size.
- [ ] I can walk the scheduler step, preemption included.
- [ ] I can explain why 2 padded chats are slower than 2 solo ones.

## Decoding speed

Each decode step reads every weight once: 1,192 MB at 101.4 GB/s is 11.8 ms, the floor. The layers took 25.5 ms. The
gap was overhead: Python walking the graph and launching hundreds of small kernels every step. Three levers exist: pay
less overhead per step (capture), read fewer bytes per token (quantization), or get more tokens per read (speculative
decoding). Capture won.

Capture records the decode step once with `mx.compile` and replays it. A recording replays only the shapes it saw, and
the old KV cache grew one position per step, so every step had new shapes (SPEC cut capture on 2026-11-16 for that
reason). The fixed-capacity cache fixes the shapes: buffers grow 256 positions at a time, each step writes its keys and
values at position n through the new `mlops.Write` primitive, and attention reads the first n positions, with n passed
in as an array. The whole step, 28 blocks plus the final norm, is one trace stored on the first cache.

Code:

- `stackgrad/attention.py:33-56`: `KVCache`: `reserve` rounds up to a multiple of 256 (line 45), `update` writes.
- `stackgrad/mlops.py:355-364`: `Write`; `stackgrad/backend/metal.py:107-109`: `slice_update` with pos as an array.
- `stackgrad/gpt.py:74-79`: when capture applies; `stackgrad/gpt.py:94-108`: `_decode` and `_step`.
- `stackgrad/backend/metal.py:88-91`: `capture`, Tensors in and out of `mx.compile`.
- `stackgrad/mlops.py:308-309`: the causal mask hides keys past n; `metal_attention.py:4`: kernels read n from a buffer.
- `stackgrad/quant.py:9-65`, `stackgrad/backend/metal_quant.py:6-44`: group-wise quantization and its kernel.
- `stackgrad/speculative.py:26-68`: speculative decoding; rollback is `c.n -= ...` (`:55-59`).

Numbers:

- Decode step through the layers: 25.49 ms capture off, 11.88 ms on, 2.15x, output gap 0.0
  (`bench/results/compile_metal.json`).
- Full step with the head and the logits copied to the host: 14.4 ms bfloat16, against the 11.8 ms floor, 82%
  (`bench/results/quant.json`).
- The 250-problem GSM8K run: 2,271.9 s before, 864.7 s after, same 152 correct and same 53,261 tokens
  (`bench/results/gsm8k_base.json`, `bench/results/gsm8k_base_final.json`).
- Quantization: weights 1,192 MB bf16, 633 MB int8, 335 MB int4; step 14.4, 21.3, 12.1 ms; GSM8K-text perplexity 4.269,
  4.281, 5.094 (int4 +19.3%) (`bench/results/quant.json`).
- Speculative, Qwen3-0.6B drafting for 1.7B: 1.7B alone 26.53 tok/s, 0.6B alone 63.7; k = 1 acceptance 0.797 at
  0.827x; best k = 4 at 0.973x; outputs identical to the 1.7B alone (`bench/results/speculative_metal.json`).

??? question "Whiteboard: why is decode memory-bound?"
    Per token, every weight is read once and used in one multiply-add: 2 FLOPs per 2-byte bfloat16 weight, 1 FLOP per
    byte. The M4's ridge point is about 36 FLOPs per byte. Bytes: 1,192 MB / 101.4 GB/s = 11.8 ms. FLOPs:
    2 x 596M = 1.19 GFLOP / 3.62 TFLOP/s = 0.33 ms. Memory time is 36x compute time, so the floor is bandwidth. Batching
    raises FLOPs per byte, which is why 8 chats cost 3.2x one.

??? question "What is capture, and why does it need fixed shapes?"
    `mx.compile` runs your Python once per input shape, records the graph of MLX ops, and later calls replay that graph
    without Python, as fused kernels. A new input shape triggers a new trace. The old cache grew by one position per
    step, so every step was a new shape and would retrace every token. With fixed-capacity buffers the step's inputs keep
    their shapes until a buffer grows. What changes per step (the position n, the RoPE cos and sin) goes in as array
    inputs, not Python constants (`gpt.py:100-101`), so the same trace serves every position.

??? question "Whiteboard: why does the cache grow in 256-position chunks?"
    Growing a buffer copies it and changes its shape, which recompiles the trace, measured at about 20 ms
    (`attention.py:40-41`). Per-token growth would put that on every 12 ms step. Growing by 256 spreads 20 ms over 256
    tokens, 0.08 ms per token (arithmetic). The memory cost is at most 255 unused positions, 255 x 114,688 bytes = 29 MB
    per sequence (arithmetic). Allocating the full 4,096-token context up front would hold 470 MB per sequence whether
    it's used or not. 256 sits between. The rounding is to a multiple of 256: a 99-token prompt gets capacity 256 and a
    261st token grows it to 512, which I checked on CPU.

??? question "What does mlops.Write do, and why make it a primitive?"
    It returns `buf` with x written along axis 2 starting at `pos` (`mlops.py:355-364`). On NumPy it writes in place;
    on MLX it calls `slice_update` with pos as an int32 array (`metal.py:107-109`), so inside a trace pos is an input
    and the trace stays valid for every position. As a primitive it fits the lazy graph: its `out_shape` is the buffer's
    shape, so later ops record before it runs. It has no backward, which is fine because caches only fill under
    `no_grad`. I checked on CPU that the NumPy path reuses the same buffer object across writes.

??? question "When does a decode step run captured, and when doesn't it?"
    `hidden()` captures when the backend has `capture` (Metal), the caches already hold tokens, there's no padding, and
    gradients are off (`gpt.py:77`). So prefill isn't captured, and neither are padded batches: batched serving of
    different-length chats and GRPO's left-padded rollouts run eager. Multi-token steps on a warm cache are captured,
    each shape with its own trace: on a tiny Metal model I checked that a 5-token step goes through `_decode`. Each
    `generate()` call builds new caches, so it compiles a new trace on its first step, since the trace lives on the
    first cache (`gpt.py:98-99`).

??? question "Why does the trace live on the first KV cache?"
    `mx.compile` bakes the model's weights into the trace as constants, and the trace's input shapes are the cache
    buffers' shapes. Keeping it on `caches[0]` ties its lifetime to the caches it was built for: when a generation's
    caches go away, so does its trace (`gpt.py:92-93`).

??? question "Explain the decode kernel rewrite and why it mattered."
    After the fixed cache landed, a step was still about 20 ms, and the old thread-per-row decode kernel ran 16 threads
    for a 16-head, one-token step. The rewrite gives each query row a threadgroup of D = 128 lanes, which score 128 keys
    in parallel per tile, then run the online softmax over the tile in key order so the result rounds like the old
    kernel. With it the captured step reached 11.9 ms. The agent measured 28 dependent calls at 80 keys taking 1.1 ms
    instead of 10.3 (session log; no result file).

??? question "Whiteboard: group-wise int4 quantization."
    Take 64 weights, find min and max, and lay 16 equally spaced levels between them: `scale = (max - min) / 15`. Store
    each weight as `round((w - min) / scale)`, 4 bits, packed 8 to a uint32, plus a scale and a bias (the min) per group
    (`quant.py:9-19`). Dequantize as `q * scale + bias`. The kernel avoids dequantizing every weight: per packed word it
    computes `sum(x * q)` and `sum(x)`, then applies `scale * sum(x q) + bias * sum(x)` once per word
    (`metal_quant.py:18-24`). Bytes: 1,192 MB to 335 MB.

??? question "int4 cut the bytes 3.6x. Why did the step only go from 14.4 to 12.1 ms?"
    Reading 335 MB at 101.4 GB/s takes 3.3 ms. The rest of the 12.1 ms is work the weight bytes don't explain: overhead
    the trace didn't remove, attention, the head's logits going to the host. Quantization attacks the bytes, and after
    capture the bytes were no longer most of the step.

??? question "Why is int8 slower than bfloat16?"
    21.3 ms vs 14.4. MLX's own int8 kernel gives the same 21.3 ms (`bench/results/quant.json`), so it isn't my kernel. I
    haven't profiled where the time goes. That's the honest answer, and the next step is a per-kernel profile of one
    int8 step.

??? question "Why does speculative decoding preserve the target's distribution?"
    The draft proposes x with probability q(x). Accept it with probability `min(1, p(x)/q(x))`; on rejection, sample from
    `max(0, p - q)` normalized (`speculative.py:48-52`). The probability of emitting x is then
    `q(x) min(1, p/q) + (1 - sum_y min(p, q)) * max(0, p(x) - q(x)) / Z`, which works out to p(x). If all k drafts are
    accepted, the target's (k+1)-th distribution gives a bonus token. With greedy decoding the outputs match the 1.7B
    alone, token for token, in every row of the results file.

??? question "Why does speculative decoding now lose (0.97x at best)?"
    Capture made the target fast, so the draft looks expensive: its step costs 42% of the target's (26.53 / 63.7). At
    k = 1 each round writes 1 + 0.797 = 1.80 tokens on average. If the verify pass cost one target step, a round would
    cost 1.42 steps and speculation would win 1.27x (arithmetic). Measured 0.827x means a round costs about 2.17 target
    steps, so the verify pass and the per-token host round trips (each draft token's logits go to NumPy to sample) cost
    about 1.8 steps. The verify pass has T = k + 1 rows, so it runs v3, whose 32-row threadgroups leave most rows idle, and it
    compiles its own trace. I haven't profiled the split between those. Before capture, k = 1 gained 1.068x
    (`git show HEAD:bench/results/speculative_metal.json`).

??? question "Why did capture beat quantization and speculative decoding?"
    Because overhead was most of the step: 25.5 ms against an 11.8 ms weight read. Quantization and speculation both
    attack weight reads. Removing the overhead got 2.15x; int4 on top got another 14.4 to 12.1 ms.

- [ ] I can do the memory-bound arithmetic and explain the 82%.
- [ ] I can explain capture, fixed shapes, the 256-chunk rule, and `Write`.
- [ ] I can explain why int4 and speculation underdeliver after capture.

## Fine-tuning: LoRA, GRPO, RFT

All three runs train a LoRA adapter on Qwen3-0.6B and score it on the same protocol: the first 250 GSM8K test problems,
greedy decoding, at most 320 new tokens, solved when the final number matches (`bench/gsm8k.py`). The released model
solves 152.

LoRA learns `W + (alpha/r) A B` with A (in, r) random and B (r, out) zero, so training starts from the unchanged model.
GRPO samples a group of replies per problem, scores each 1 or 0, and pushes each reply up or down by how it compares to
its siblings. RFT samples the model's own replies, keeps the right ones that stop, and fine-tunes on them. The decision
across all three: compare paired, problem by problem, with a sign test, because totals hide churn.

Code:

- `stackgrad/lora.py:10-37`: `LoRALinear` and `apply_lora`; `:44-54`: `merge_adapter`.
- `train_lora.py:26-33`: the mask over solution tokens; `:36-40`: the head only where the loss looks; `:97-103`:
  micro-batches weighted by reply-token share.
- `train_grpo.py:22-24`: the reward; `:27-36`: left-padded group sampling; `:39-41`: `group_advantages`;
  `:44-51`: `policy_loss`; `:105-116`: skip zero-advantage replies, no step without gradient.
- `scripts/rft_sample.py:15-69`: RFT sampling with the GRPO reward as the filter.
- `bench/gsm8k_compare.py:9-12`: the exact two-sided sign test.

Numbers:

| Model | Solved | Fixed / broke vs released | p | Avg reply | Hit 320 |
|---|---|---|---|---|---|
| released | 152 | | | 213 | 45 |
| LoRA on GSM8K | 123 | 23 / 52 | 0.0011 | 92 | 11 |
| GRPO, reward ignores stopping | 154 | 31 / 29 | 0.90 | 320 | 250 |
| GRPO, reward requires stopping | 139 | 25 / 38 | 0.13 | 216 | 29 |
| RFT | 155 | 18 / 15 | 0.73 | 208 | 37 |

Sources: `bench/results/gsm8k_*.json` and `bench/results/gsm8k_base_vs_*.json`. Training: LoRA loss 0.770 over the first
50 steps to 0.573 over the last 50 (`runs/lora_gsm8k/metrics.jsonl`); GRPO reward 54% over the first 20 steps to 67%
over the last 20, cut-off share 27% to 8%, 38.8 s sampling and 7.3 s updating per step (`runs/grpo_gsm8k/metrics.jsonl`);
RFT sampled 3 replies for each of 3,328 problems, 60% right, 5,991 kept at 180 tokens average, loss 0.221 to 0.216
(`runs/rft_gsm8k/config.json`, `runs/rft_gsm8k/metrics.jsonl`).

??? question "Whiteboard: count LoRA's trainable parameters for Qwen3-0.6B at rank 16."
    Each adapted matrix of shape (in, out) adds `r (in + out)`. Per layer: q_proj 1,024 to 2,048: 16 x 3,072 = 49,152;
    k_proj and v_proj 1,024 to 1,024: 32,768 each; o_proj 2,048 to 1,024: 49,152; w1 and w3 1,024 to 3,072: 65,536
    each; w2 3,072 to 1,024: 65,536. Sum 360,448 per layer, times 28 layers = 10,092,544, 1.69% of 596M. Adam's two
    moments then cost 81 MB in float32 instead of 4.8 GB for a full fine-tune (arithmetic).

??? question "Why is B zero and A random, and not both zero?"
    B = 0 makes `AB = 0`, so step 0 is the released model. If A were zero too, B's gradient `(xA)^T g` would be zero and
    A's gradient `x^T (g B^T)` would be zero: nothing would ever move. A random A gives B a gradient on the first step.
    A is scaled by `1/sqrt(n_in)` and cast to float32 after the division (`lora.py:16-17`), which is the float64 story.

??? question "Why does LoRALinear scale the rank-r middle and not the output?"
    `(x A * scale) B` multiplies an (N, 16) array; scaling the output would multiply an (N, out) array, up to 3,072 wide.
    Same math, a smaller copy (`lora.py:24`).

??? question "Why did LoRA make the model worse at GSM8K?"
    It learned GSM8K's house style along with its format: replies fell from 213 to 92 tokens. The released model writes
    out its steps, and those steps were getting answers right. Training loss fell from 0.770 to 0.573 on the solution
    tokens while test accuracy fell from 60.8% to 49.2%. Problem 56 shows it: the released model counts ticket plus
    popcorn, $14 a visit, 3 visits; the LoRA model drops the popcorn and answers 6.

??? question "Whiteboard: GRPO advantages for rewards 1, 0, 0, 0."
    `A_i = (R_i - mean(R)) / std(R)`. Mean 0.25. Population std `sqrt(0.25 x 0.75) = 0.433`. The right reply gets
    0.75 / 0.433 = 1.73; each wrong one -0.25 / 0.433 = -0.58. The pushes sum to zero: 1.73 = 3 x 0.58. A group of all
    1s or all 0s has std 0, so every advantage is 0 (the code adds eps = 1e-6 to the std), and the step skips those
    replies (`train_grpo.py:39-41, 105-106`). I checked both cases on CPU.

??? question "Why divide by the 320-token budget and not by each reply's length?"
    Dividing by length gives each token of a long reply a smaller weight. A long wrong reply then gets a weaker push down
    per token, so length becomes a way to dodge the penalty. Dividing every reply by the same budget gives every token the
    same weight (Dr. GRPO, Liu et al. 2025) (`train_grpo.py:45-51`).

??? question "Where's the PPO ratio, the clipping, the KL penalty?"
    The run is on-policy with one update per batch of rollouts, so the importance ratio between the sampling policy and
    the updated policy is 1 at the update, and clipping never triggers. The loss reduces to REINFORCE with a
    group-normalized baseline. There's no KL penalty to the reference model either, and I didn't measure how far the
    policy drifted from the base in 200 steps. For a longer run, the KL term is the first thing I'd add.

??? question "Why sample all 64 replies in one batch?"
    Decode is weight-bound, so rows are nearly free: on the 3090 a decode step takes 79.0 ms for one row and 86.4 ms for
    64, which is 12.7 against 740.4 tokens per second (`bench/results/rollout_cuda.json`). The 64 rows come from 8
    problems x 8 replies, left-padded so each row's real tokens end at the same column (`train_grpo.py:28-33`).

??? question "What is RFT, and why was there little left to learn?"
    Rejection-sampling fine-tuning, or expert iteration: sample replies to training problems, keep the ones with the
    right number that end on the end-of-turn token, dedupe, and fine-tune on them like ordinary examples. Every example
    is the model's own writing, so its style can't drift toward GSM8K's. The loss started at 0.221 and ended at 0.216:
    the model already rated its own replies as likely, so supervised training on them had little gradient. Replies
    stayed long, 208 tokens, and the result was 155 against 152, inside the noise.

??? question "Whiteboard: run the sign test on GRPO's 25 fixed, 38 broke."
    Only changed problems carry information. If training changed nothing, each of the 63 is a fair coin. The two-sided
    p-value is `2 x P(X <= 25)` for X ~ Binomial(63, 0.5), which is 0.13 (`gsm8k_compare.py:9-12`). So 139 vs 152 isn't
    a demonstrated drop. LoRA's 23 vs 52 gives 0.0011, a real one.

??? question "Why a paired test instead of comparing 60.8% and 55.6%?"
    Both runs answer the same 250 problems, and most problems are solved by both or neither. A two-proportion test treats
    the samples as independent and throws that pairing away, so it sees more noise than there is. The sign test uses
    only the discordant pairs. It also catches churn: 139 vs 152 hides 25 fixes and 38 breaks.

??? question "How did activation checkpointing change LoRA's memory?"
    On the 3090 with 8 examples per batch: without checkpointing, micro-batch 2 peaked at 10.0 GB in 0.79 s per batch,
    micro-batch 4 at 15.16 GB in 0.40 s, and micro-batch 8 ran out of memory. With checkpointing, micro-batch 8 peaked at
    5.23 GB in 0.42 s (`bench/results/checkpointing_cuda.json`). Recomputing each block in backward bought a bigger
    micro-batch, which bought back the speed.

- [ ] I can count LoRA parameters and explain the init.
- [ ] I can compute GRPO advantages and explain budget normalization.
- [ ] I can run the sign test and defend pairing.

## RAG and the agent

Both apps are programs that call the server's OpenAI-style endpoint, with stackgrad computing every number from the
embedding to the last sampled token. RAG embeds 2,067 SQuAD paragraphs with Qwen3-Embedding-0.6B (Qwen3's architecture
with its own weights, so the loader already runs it), searches by dot product, and puts the top 3 in the prompt. The
agent gives Qwen3 a calculator through its native tool-calling format and runs a ReAct loop.

Code:

- `apps/rag.py:23-36`: embedding, last-token pooling, unit length; `apps/rag.py:101`: the search.
- `apps/rag.py:66-73`: exact match, F1, and "contains".
- `apps/agent.py:26-47`: the calculator, an AST allowlist; `apps/agent.py:60-79`: the tool loop.
- `stackgrad/qwen3.py:90-108`: tool definitions and tool results rendered as Hugging Face's template does.

Numbers (`bench/results/rag.json`, 200 questions): recall@1 73.0%, recall@3 91.5%; answer contains the reference 3%
closed-book vs 55% with paragraphs; exact match 0% vs 20.5%; F1 0.085 vs 0.400; 0.48 s vs 1.53 s per answer; 61.3 ms to
embed a question; 362.6 s to build the index. Agent (`bench/results/gsm8k_agent*.json` and the compare files): with the
"step by step" prompt, 0 tool calls and 132 of 250 (21 fixed, 41 broke, p = 0.015); with the tool requested in the user
turn, 440 calls on 248 problems and 42 of 250 (7 fixed, 117 broke).

??? question "Walk me through retrieval end to end."
    Offline, embed every paragraph: run Qwen3-Embedding over the text plus an end-of-text token, take the final hidden
    state at the last token, and scale it to length 1 (`rag.py:23-36`). Stack them into a (2,067, 1,024) matrix. Online,
    embed the question with Qwen3-Embedding's instruction prefix, take `index @ query`, which is cosine similarity
    because both sides have unit length, and keep the top 3. The server then answers with those paragraphs in the
    prompt.

??? question "Why last-token pooling?"
    The model is causal, so only the last position has attended to the whole text. Qwen3-Embedding was trained with that
    pooling, and stackgrad's embeddings match Hugging Face's to 5.5e-5 (SPEC). Batches are left-padded so every row's
    last token sits in the last column (`rag.py:33-34`).

??? question "Whiteboard: what does one search cost, and would you add an ANN index?"
    2,067 paragraphs x 1,024 dimensions = 2.1M multiply-adds, about 4 MFLOP (arithmetic). That's microseconds on any
    hardware, against 61 ms to embed the question. An approximate nearest-neighbor index buys nothing at this size. At
    millions of paragraphs the brute-force product would cost billions of operations per query, and then I'd reach for
    one.

??? question "Why score 'contains' instead of exact match?"
    Qwen3 answers in full sentences, so an exact-match score calls a right answer wrong: with paragraphs, exact match is
    20.5% while "contains the reference" is 55%. I report all three: exact match, token F1, and contains.

??? question "Why is closed-book a floor and not a baseline?"
    SQuAD's questions were written by people reading the paragraph, so many make no sense without it ("what did the
    committee decide?"). Part of the 3% closed-book score comes from questions that assume context the model never saw,
    so it bounds the retrieval gain from below.

??? question "How is the calculator safe to run on model output?"
    It parses the expression with Python's `ast` and evaluates only numbers and `+ - * / ** // ( )` from an allowlist,
    never `eval` (`agent.py:26-47`). Constants become floats, because int powers grow without bound and floats overflow
    to an error. Exponents over 64 are refused. Every error comes back to the model as text instead of crashing the
    loop.

??? question "Why did the calculator make the model worse?"
    With the "solve it step by step" prompt, the model never called the tool, and the tool's presence in the system
    prompt still shortened replies from 213 to 148 tokens: 132 vs 152. When the user turn asked for the tool, it called
    it on 248 problems, but it handed the calculator arithmetic before working out what to compute, then answered with
    whatever came back: 42 solved, replies averaging 27 tokens. The worked steps were what got problems right, and
    asking for the tool pushed them out.

??? question "How did you rule out a bug in your own stack?"
    I ran the same prompt through Hugging Face's own model. It ignored the tool too, because the prompt said "solve it
    step by step" and the model followed that. One check against the reference saved a long search.

??? question "Where does the agent's time go?"
    911.9 s in 500 model calls against 0.018 s in 440 calculator calls (`bench/results/gsm8k_agent.json`). The tool is
    free; every round trip to the model costs a prefill of the whole conversation so far, 155,000 prompt tokens in all.

- [ ] I can explain embedding, pooling, and the search, with the cost arithmetic.
- [ ] I can explain both agent runs and why the second collapsed.

## Debugging stories

Tell each one in the order situation, what you saw, how you narrowed it, root cause, fix. Interviewers care most about
the narrowing.

??? question "bqa3's bad VRAM"
    **Situation.** On 2026-11-21 I was verifying the realize memory fix on the RTX 3090: a 16-row, 96-token Qwen3-0.6B
    prefill, old realize vs new, both compared against a CPU reference.

    **What I saw.** The new path matched the CPU within 0.00488 at hidden-state magnitudes around 7,400, which is float32
    rounding. The old path, which holds every intermediate, was off by about 1.06 at the output. Since the old code was
    what all earlier CUDA runs used, that was alarming.

    **How I narrowed it.** I recorded each block's output. Both paths sat at 0.00488 through block 9; the old path jumped
    to 0.363 at block 10, 0.631 at block 11, 3.62 at block 14. Suspecting the Triton kernel's partial last tile (96
    isn't a multiple of 64), I reran with Triton off: still wrong (0.851 at block 10). With launch blocking: still
    wrong, so not a race. Then I logged a fingerprint of all 2,332 ops' outputs in both modes, the sum and the absolute
    sum, and diffed them. The first mismatch was op 869, a `Mul` computing `x * x` inside block 10's qk-norm. Its
    output is non-negative, so its sum must equal its absolute sum. In the held run the absolute sum came out smaller
    than the sum, 4,608,285.609 against 4,608,285.665. That can't happen in arithmetic. The same buffer had returned
    different values on two consecutive reads.

    **Root cause.** The card's memory. A test that filled 20 GB of the 23.6 GB with a distinct pattern per GB and read
    it back five times found 1,462 wrong 32-bit words, identical on every pass, all in the 5th to 9th GB allocated, with
    the card at 38 C. A consumer card has no ECC, so nothing logs it. The old realize held about 9 GB (2.38 GB of
    weights plus 6.85 GB of intermediates), reaching into the bad region; the new one peaked 0.19 GB above the weights.

    **Fix.** No software fix exists. CUDA work for the project is paused until the card is tested with a proper tool
    (cuda_memtest or gpu-burn) or replaced. Past CUDA training runs are noted, not rerun; every GSM8K score was measured
    on the Mac. The lesson: an invariant that must hold (|sum| <= sum of |x|) separates a software bug from hardware
    faster than any amount of reading code.

    Sources: the 2026-11-21 session log for the per-block and per-op numbers; `CLAUDE.md` and the project memory for
    the read-back test. No result file holds these.

??? question "realize holding every intermediate"
    **Situation.** RFT sampling on the 3090 ran out of memory. The agent running it measured a 16-row x 96-token prefill
    at 12.3 GB, against 3.1 GB when each block realized itself, and patched every block to realize on its own as a
    workaround (`scripts/rft_sample.py:32-35`).

    **What I saw.** Peak memory scaled with the size of the graph, not with what was alive at any moment. Laziness had
    introduced it: eager code frees an intermediate when its last Python reference dies.

    **How I narrowed it.** I read `realize`. Its `order` list, built by the DFS, referenced every node in the graph until
    the function returned, and each node referenced its buffer once stored.

    **Root cause.** The scheduler's own bookkeeping kept every buffer alive for the whole run.

    **Fix.** Delete `order`, `seen`, and `run_of` before executing, drop each run from the list as it executes, and let
    `_store` drop `_lazy` so a buffer dies when its last reader has run (`tensor.py:170-172`, `tensor.py:97`). On CUDA
    the pool peak above the weights went from 6.85 GB to 0.19 GB with the same output (session log). On Metal nothing
    changed, 3.71 GB both ways (`bench/results/realize_memory_metal.json`), because MLX's own laziness means the GPU work
    happens in `mx.eval` after realize has returned and dropped its locals. `bench/realize_memory.py` reproduces the
    comparison on either device.

??? question "MLX buffer-cache bloat slowing 4-chat serving"
    **Situation.** After the realize fix I reran the serving benchmark. The batched 4-chat row ran at 21.9 tok/s, 87.57
    ms per step, while 2 chats ran 51.6 and 8 chats 107.5. SPEC already listed the 4-chat row as an open oddity.

    **What I saw.** One row out of line, and the memory fix hadn't touched it, so it wasn't stackgrad holding buffers.

    **How I narrowed it.** MLX keeps freed buffers in a cache for reuse. At that time the KV cache grew by
    concatenation every step, so freed buffers came in ever-new sizes the cache couldn't reuse. One experiment: cap the
    cache with `mx.set_cache_limit(1 << 30)` in a one-off run of the benchmark. 4 chats went to 92.2 tok/s and 8 chats
    to 138.1.

    **Root cause.** The allocator's cache, not live arrays, was eating memory and slowing the run.

    **Fix.** The cap went into the Metal backend (`metal.py:84`), and the rerun gave 91.9 tok/s at 4 chats and 137.9 at
    8 (session log). After the fixed-capacity cache landed, the cap matters less: 8-chat serving holds 2.4 GB cached
    uncapped vs 1.1 GB capped at the same speed (`metal.py:84` comment), and today's file shows 96.3 and 154.9
    (`bench/results/serve_batch_metal.json`). Stable Diffusion had the same shape earlier: 5.6 GB of live arrays in a
    12.4 GB footprint, 6.0 GB after capping, same speed (`docs/guide.md`). Watch the process footprint, not only your
    arrays.

??? question "The float64 LoRA run"
    **Situation.** The first LoRA run on GSM8K took 136 minutes for 935 steps.

    **What I saw.** Throughput around 165 tokens per second (`runs/lora_gsm8k_f64.log`), and it fit only one example per
    pass. The fixed run later did 935 steps in 890 s, 14.8 minutes, at micro-batch 2 (`runs/lora_gsm8k.log`).

    **How I narrowed it.** Shapes were right, so I checked dtypes: activations were float64 from the first adapted layer
    on, forward and backward.

    **Root cause.** A was initialized as random float32 numbers divided by `np.sqrt(n_in)`. Under NumPy 2's promotion
    rules, `np.sqrt` returns a float64 scalar, and float32 array / float64 scalar gives float64. (A plain Python float
    would have stayed float32.) stackgrad keeps float dtypes as given, so x (float32) @ A (float64) promoted every later
    layer. The 3090 runs float64 at 1/64 of its float32 rate, and every activation took 8 bytes instead of 4.

    **Fix.** Cast after dividing (`lora.py:16-17`), and `tests/lora.py` now checks that the adapter and the loss stay
    float32. The float64 adapter solved 124 against 123 for the float32 one (`bench/results/gsm8k_lora_gsm8k_f64.json`),
    so the bug cost time, not accuracy. Watch dtypes like shapes.

??? question "The no-RMSNorm run going NaN at step 627"
    **Situation.** The CS336 ablation removes every RMSNorm and trains at the baseline's learning rate, 1e-3.

    **What I saw.** In `runs/abl_no_rmsnorm/metrics.jsonl`: step 621 loss 2.65; step 622 loss 27.1 with gradient norm
    10,532; step 623 back to 2.66; step 624 5.8 (norm 494); step 625 84.1 (norm 155,671); step 626 loss 8.1e8 and a
    non-finite norm; step 627 NaN, and NaN from then on.

    **How I narrowed it.** The spikes come and go before the blow-up, which looks like an instability, not a data bug.
    The same config at lr 3e-4 trained to 1.699 (`bench/results/ablations.json`), so the learning rate is the lever.

    **Root cause.** Without normalization nothing rescales the residual stream between layers, so a large update grows
    the activations, which grows the next gradient. Clipping at 1.0 doesn't stop it: AdamW's step is `m / sqrt(v)`,
    about lr per weight whatever the gradient's scale, so clipping the gradient does little to the update. Once logits
    get large enough, exp overflows to inf and inf - inf gives NaN.

    **Fix.** For the ablation, none: divergence is the result. It's the strongest argument in the project for pre-norm,
    and a lower learning rate is the workaround.

??? question "The GRPO reward hacking"
    **Situation.** The first GRPO run rewarded the right final number, nothing else.

    **What I saw.** From step 51 on, every sampled reply ran to the 320-token limit (`runs/grpo_no_stop/metrics.jsonl`).
    The reward stayed fine.

    **How I narrowed it.** I read the samples. The model wrote its solution, then "The final answer is `\boxed{18}`."
    between rows of dashes, over and over. My answer extractor reads the first `####` or `\boxed{}` answer
    (`bench/gsm8k.py:19-22`), which stayed right, so the reward couldn't tell, and writing on cost nothing.

    **Root cause.** The reward I wrote had a loophole, and the policy found it. The adapter even scored 154 of 250 on the
    test set, all 250 replies hitting the limit, with a sign test of 31 fixed and 29 broke, p = 0.90: no real change.

    **Fix.** Make stopping part of the answer: reward 1 only if the reply ends with Qwen's end-of-turn token and the
    number is right (`train_grpo.py:22-24`). The cut-off share fell from 27% to 8% over training. The lesson: when an
    objective improves and the thing you care about doesn't, read the outputs.

- [ ] I can tell each story in under two minutes, with the narrowing steps in order.

## Trade-offs and weak spots

??? question "Why only one GPU? Where's distributed training?"
    Both models fit on one 3090, so data parallelism, optimizer sharding, and FSDP were cut on purpose (SPEC). I can
    describe what I'd add: data parallel first, an all-reduce of gradients after backward overlapped with the backward
    of earlier layers, then ZeRO-style sharding of Adam's state when it stops fitting. I haven't built or measured it,
    and I say so.

??? question "Why a 3,000-line budget?"
    Every line is a maintenance cost, and the budget forces deletion before addition. CI fails when the package outgrows
    it (`scripts/lines.py --max`); it sits at 2,977 (`bench/results/lines.json`). It made real decisions: v2 of the Metal
    kernel was deleted to pay for the fixed-capacity cache. It also leaves shortcuts, each marked in the code with when
    to upgrade.

??? question "You have laziness but no codegen. Isn't that half a compiler?"
    Yes, on purpose. Laziness cost 0.7 ms per step on Metal and bought nothing alone (`bench/results/lazy.json`). Fusion
    through `mx.compile` bought 1.02x on prefill. Graph capture bought 2.15x on decode. The measurements say overhead was
    the problem, not elementwise memory traffic, so capture came before codegen. On CUDA `fuse` is still the identity
    (`backend/cuda.py:24`); CUDA graphs are the next step there, paused with the card.

??? question "Padded batches run outside the trace. Isn't that the case that matters for serving?"
    It is a real gap. Batched serving of different-length chats and GRPO rollouts carry left padding, and padding skips
    capture and the fused attention kernels. That's why 2 batched chats run slower than 2 solo ones (52.4 vs 64.8
    tok/s). The fix is to make padding an input: per-row lengths into the decode kernel and the mask, so a padded step has
    fixed shapes too. A paged decode kernel that reads block tables would also remove the cache copies on every
    membership change.

??? question "Your CUDA training runs ran on a card with bad memory. Can you trust them?"
    Partly. The bad region starts around the 5th GB allocated, so runs that stayed under that are unaffected, and runs
    above it could have read wrong values with no error raised. The fine-tuned adapters were all scored on the Mac, so their GSM8K
    numbers describe the adapters as trained. The TinyStories run's loss curve is plausible and the model writes
    coherent stories, but I can't prove no value was corrupted. I'd rerun on a tested card before publishing the CUDA
    numbers as anything stronger.

??? question "Your fine-tuning didn't improve the model. Why put it on a resume?"
    Because the engineering and the measurement are right, and the result is honest. I fixed the eval protocol before
    training, compared paired with a sign test, found the reward hack by reading samples, and reported a real drop
    (LoRA, p = 0.0011) and two null results. An eval of 250 problems with one seed can only detect large effects; with
    more time I'd evaluate the full test set and several seeds.

??? question "What would you do differently?"
    Three things. Build a paged decode kernel early, so serving never copies caches and padded batches can run captured.
    Save optimizer state and data position in checkpoints from day one. Test new GPU hardware with a fill-and-read-back
    before trusting it.

??? question "What are the sharp edges you know about?"
    - Each `generate()` call compiles a fresh trace on its first decode step (`gpt.py:98-99`), and so does each new
      batch composition in the runner that runs captured (one chat, or equal lengths), since `_join` builds new caches
      (`runner.py:48`).
    - Speculative decoding now loses to the target alone (0.97x at best).
    - int8 decodes slower than bfloat16 (21.3 vs 14.4 ms), unexplained.
    - Metal has no fused attention backward, so attention backward costs 14.5x its forward.
    - `col2im` (convolution backward) round-trips through the CPU on Metal and CUDA (`backend/metal.py:98`).
    - `Write` has no backward; a cache write under gradient tracking would fail at `backward()`.

- [ ] I can answer each weak-spot question without getting defensive.

## Using AI coding tools

Be direct about it. You directed the project: you chose the scope, the order, the constraints (the line budget,
tinygrad's style, "every claim comes with a measured number"), and the tests. You verified every claim against
measurements, and you published unflattering numbers when they came out. Agents wrote some of the code, including
everything from 2026-11-21 listed below. Say "I used agents, and here is how I checked them" before the interviewer has
to ask; two follow-up questions would surface it anyway.

??? question "How much of this did you write yourself?"
    The architecture, the autograd, the backends, the kernels through v3, the training and fine-tuning experiments, and
    the evaluation protocol are mine, built and debugged over the project. On the last day I used AI agents for a batch
    of performance work: graph capture with a fixed-capacity KV cache, the decode kernel rewrite, fusion in realize, the
    loader speed-up, and a few fixes. I wrote their briefs with the constraints and the measurement each had to report,
    reviewed every diff, and checked the results: the 250 GSM8K replies came out identical word for word with capture
    on, and every test in the suite passes. I can walk you through any of those lines.

??? question "How do you verify code an agent wrote?"
    Same as code I wrote, with less trust. A before/after measurement on the real workload, interleaved so drift hits
    both sides. A correctness check against a reference: token-identical outputs, bitwise-equal kernels, the full test
    list. And reading the diff line by line until I can explain each line, because an agent's report is a claim, not
    evidence.

??? question "Tell me about a time an agent's output needed correcting."
    The KV-cache agent went beyond its brief: it rewrote the v1 decode kernel and deleted v2 to stay under the line
    budget. Both turned out right, but it left the post's kernel table stale, since the results file still has the old
    v1 and v2 rows and no row for the new v1. Preparing for interviews I also found a sentence in the post saying
    speculative decoding's verify pass runs outside the recording. Reading the code, it runs through the captured path
    with its own trace, and a check on a small Metal model confirmed it. The code wins; the post needs a correction.

### Agent-written diffs to study line by line

Read each with `git diff HEAD -- <file>` until you can explain every changed line.

- [ ] `stackgrad/attention.py:9-56`: `rope_tables`, `apply_rope(x, cos, sin)`, `KVCache` with `reserve` and `update`.
- [ ] `stackgrad/attention.py:77-102`: positions from the cache inside the trace, the slice to n outside it.
- [ ] `stackgrad/mlops.py:301-352`: `Attention` takes n; the causal mask hides unwritten slots.
- [ ] `stackgrad/mlops.py:355-364`: `Write`.
- [ ] `stackgrad/gpt.py:74-108`: `hidden`'s capture condition, `_layers`, `_decode`, `_step`.
- [ ] `stackgrad/backend/metal.py:84-91`: cache cap, `fuse`, `capture`; `:107-118`: `write`, `fused_attention(n)`.
- [ ] `stackgrad/backend/metal_attention.py:3-33`: the threadgroup-per-row v1, n from a buffer; `:133-156`: v2 removed.
- [ ] `stackgrad/backend/cuda.py:24, 71-73` and `stackgrad/backend/cpu.py:79-81`: `fuse` as identity, n on CUDA.
- [ ] `stackgrad/tensor.py:81-124`: `_FUSABLE`, `_fusable`, `_store`, `_realize_fused`, `_fused`.
- [ ] `stackgrad/tensor.py:149-184`: realize's runs, `needed`, and freeing.
- [ ] `stackgrad/tensor.py:346`: the `requires_grad` assertion.
- [ ] `stackgrad/layers.py:10-14`, `stackgrad/gpt.py:63-64`, `stackgrad/qwen3.py:44-48`: SKIP_INIT.
- [ ] `stackgrad/serve/runner.py:36-55`: join and split with `KVCache(k, v, n)`.
- [ ] `stackgrad/speculative.py:55-59`: rollback by moving n.
- [ ] `bench/compile.py`, `bench/realize_memory.py`, and the new checks in `tests/lazy.py` and `tests/backends.py`.

## Numbers cheat sheet

| Number | What it measures | Source |
|---|---|---|
| 1.8e-15 | stackgrad vs PyTorch loss at step 1, float64 | `bench/results/m1_parity_f64.json` |
| 52 / 52 | ops agreeing with NumPy on Metal | `bench/results/backends_metal.json` |
| 4.8 vs 4.09 ms | lazy vs eager training step, Metal, small model | `bench/results/lazy.json` |
| 44.7 s | BPE training, 10,000-token vocabulary, TinyStories | `bench/results/bpe.json` |
| 22,696,448 | story model parameters | post; checked by building the model |
| 9.2335 | TinyStories loss at step 1 (ln 10,000 = 9.21) | `runs/tinystories/metrics.jsonl` |
| 1.399, 71 min | TinyStories final validation loss and wall time | `runs/tinystories/metrics.jsonl` |
| 8.5 TFLOP/s, 23.9% | median throughput and MFU vs 35.6 TFLOP/s rated | `runs/tinystories/metrics.jsonl`, `config.json` |
| +0.013 / +0.03 / +0.085 | post-norm / SiLU / NoPE vs baseline 1.524 | `bench/results/ablations.json` |
| step 627 | no-RMSNorm run goes NaN at lr 1e-3 | `runs/abl_no_rmsnorm/metrics.jsonl` |
| 3.1 / 7.6 / 42 us | 1,000-element multiply on NumPy / Metal / CUDA | `bench/results/backend_overhead_*.json` |
| 33.9 / 3.62 TFLOP/s | matmul peak, 3090 TF32 / M4 bfloat16 | `bench/results/backend_overhead_*.json` |
| 25.1x | 3090 vs M4 CPU training step | `bench/results/backend_overhead_*.json` |
| 82% | stackgrad vs plain MLX training step on Metal | `bench/results/backend_overhead_cpu_metal.json` |
| 6.7x | Metal v3 vs mx.fast.sdpa, story-model shape (1.49 vs 9.94 ms) | `bench/results/flash_attention.json` |
| 2.6x | mx.fast.sdpa vs v3 at Qwen3's 128-wide heads | `bench/results/flash_attention.json` |
| 8.5x | Triton FA2 forward + backward vs composed (0.764 vs 6.477 ms) | `bench/results/flash_attention_cuda.json` |
| 14.5x | attention backward vs forward on Metal | `bench/results/profile_metal.json` |
| 5.1e-5 / 9.7e-5 / 0.35 | Qwen3 logit gap: CPU f32 / Metal f32 / Metal bf16 | `bench/results/qwen3_parity.json` |
| 1.4e-4 / 2.2e-4 | GPT-2 logit gap, CPU / Metal, 40 greedy tokens match | `bench/results/gpt2_parity.json` |
| 114,688 B | Qwen3-0.6B KV cache per token, bfloat16 | arithmetic from `config.json` |
| 63.4 vs 154.9 tok/s | 8 chats, one pass each vs one batched pass | `bench/results/serve_batch_metal.json` |
| 25.49 to 11.88 ms | decode step through the layers, capture off vs on | `bench/results/compile_metal.json` |
| 14.4 ms vs 11.8 ms | full decode step vs weight-read floor (82%) | `bench/results/quant.json` |
| 101.4 GB/s | measured M4 memory bandwidth | `bench/results/quant.json` |
| 1.02x | fusion on a 512-token prefill | `bench/results/compile_metal.json` |
| 6.85 to 0.19 GB | CUDA prefill peak above weights, realize before vs after | 2026-11-21 session log |
| 38 to 14 min | 250-problem GSM8K run, before vs after capture | `bench/results/gsm8k_base.json`, `gsm8k_base_final.json` |
| 1,192 / 335 MB, +19.3% | bf16 / int4 weights, int4 perplexity increase | `bench/results/quant.json` |
| 0.797, 0.97x | speculative acceptance at k = 1, best speedup (k = 4) | `bench/results/speculative_metal.json` |
| 10,092,544 | LoRA trainable parameters, rank 16, 1.69% | post; arithmetic from `config.json` |
| 152 / 123 / 139 / 154 / 155 | GSM8K: released / LoRA / GRPO / GRPO no-stop / RFT | `bench/results/gsm8k_*.json` |
| p = 0.0011 / 0.13 / 0.73 | sign tests: LoRA / GRPO / RFT vs released | `bench/results/gsm8k_base_vs_*.json` |
| step 51 | GRPO without the stop requirement: every reply hits 320 tokens | `runs/grpo_no_stop/metrics.jsonl` |
| 79.0 vs 86.4 ms | 3090 decode step, 1 row vs 64 rows | `bench/results/rollout_cuda.json` |
| 136 vs 14.8 min | LoRA run in float64 by mistake vs float32 | `runs/lora_gsm8k_f64.log`, `runs/lora_gsm8k.log` |
| 73.0% / 91.5% | RAG recall@1 / recall@3 | `bench/results/rag.json` |
| 3% vs 55% | answer contains the reference, closed-book vs top-3 paragraphs | `bench/results/rag.json` |
| 132 / 42 | agent solved, step-by-step prompt vs tool requested | `bench/results/gsm8k_agent*.json` |
| 1,462 | wrong 32-bit words in bqa3's 20 GB read-back test | `CLAUDE.md`, project memory |
| 2,977 | package lines, budget 3,000 | `bench/results/lines.json` |
