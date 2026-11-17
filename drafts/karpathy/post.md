PyTorch hides the parts worth understanding. You call `loss.backward()` and gradients appear. You move a model to a GPU and it gets faster, except when it gets slower. You download a chat model and it answers, and every step between the bytes on disk and the words on screen stays out of sight.

So I rebuilt them as stackgrad, a deep learning framework in 3,350 lines of plain Python, and checked each part against the library it imitates. It starts where Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) starts, with a single number that computes its own gradient, and takes [tinygrad](https://github.com/tinygrad/tinygrad)'s design of a short list of operations with known gradients and everything else composed from them. Then it climbs the way a study plan would, through a transformer, GPU kernels, a pretrained model, a server and fine-tuning, up to retrieval-augmented generation and an agent that calls tools. Every number between your question and its answer comes from code I wrote.

TLDR:

- The gradients match PyTorch's to 1.8e-15.
- A 22.7M-parameter model reaches a validation loss of 1.399 on TinyStories in 71 minutes on one RTX 3090.
- My attention kernel runs 6.7x faster than MLX's unfused attention.
- Qwen3-0.6B's logits match Hugging Face's to 5.1e-5.
- Retrieval puts the paragraph a SQuAD question was written from in its top three 91.5% of the time, and 55% of answers then contain the reference.
- Three attempts at better grade-school math failed. LoRA lost 29 of 250 problems (p = 0.0011), GRPO changed nothing measurable (p = 0.13), and a calculator tool lost 110 (p < 0.0001).

Don't skip the last bullet.

> **Figure:** Three results, read from the result files. Left, the losses of stackgrad (thin blue) and PyTorch (wide pink) on the same model overlap step for step. Middle, the story model's validation loss over 40,000 training steps. Right, attention time at the training shape (shorter is faster), my kernel in blue, MLX's in pink (no fused kernel for heads this narrow), earlier attempts in gray.

Each section builds what the previous one showed was missing, explains its idea from scratch for a reader who knows no machine learning, and ends on a measurement. The code is at [github.com/samitmohan/minitorch](https://github.com/samitmohan/minitorch), and it ran on an Apple M4 laptop and one RTX 3090.

## Autograd

A neural network is a function with adjustable numbers inside it, the weights. Numbers go in (pixels, token ids), numbers come out, and a large language model has billions of weights. A new network starts random and gives useless answers. Learning means nudging the weights until the answers get good.

For that you need a loss, one number that says how wrong the network is (lower is better), and the gradient of the loss with respect to every weight: how much the loss moves if that weight moves a little. The gradient points uphill, so you step each weight a little downhill. That's gradient descent, and training repeats it millions of times.

The smallest network is $y=w\cdot x$. We want output $6$ for input $2$, so the loss is $L=(wx-6)^2$, which at $w=1$ is $(2-6)^2=16$. The gradient $\partial L/\partial w=2(wx-6)x=2\cdot(-4)\cdot2=-16$ is negative, so raising $w$ lowers the loss. One step with a learning rate of $0.1$ gives $w\leftarrow1-0.1\cdot(-16)=2.6$, an output of $5.2$ and a loss of $0.64$, twenty-five times smaller.

A billion weights in dozens of layers needs a program to do this, and that program is the first piece of stackgrad.

### The chain rule as a program

Any computation is a chain of small steps (multiply, add, exponentiate), and calculus has a rule for chains. If $z$ depends on $y$ and $y$ on $x$, then $\partial z/\partial x=(\partial z/\partial y)(\partial y/\partial x)$. Autograd applies that rule to every step of a program.

The forward pass runs the computation and records each step and its inputs as a graph. The backward pass walks that graph from the loss back, starting from a gradient of 1. Each step multiplies the gradient from above by its local derivative and passes the result to its inputs, and a value that feeds several steps sums what comes back. That's [backpropagation](https://www.nature.com/articles/323533a0), and the punchline is the price: one backward pass costs about two forward passes, for any number of weights.

> **Figure:** The autograd engine on stackgrad's server, values in blue and gradients in pink. At $x=2,w=-0.5,b=3$ the chain rule gives $\partial/\partial x=-2$ and $\partial/\partial w=8$. Pick another expression, or run backward again.

### Nineteen operations

A stackgrad tensor remembers the operation that produced it. An operation is a small class whose `forward` computes the result and whose `backward` takes the result's gradient and returns one per input. For $z=xy$ that's $\bar{z}\cdot y$ for $x$ and $\bar{z}\cdot x$ for $y$, where $\bar{z}$ is the gradient arriving at $z$. Condensed from stackgrad/mlops.py:

```python
class Mul(Function):
    def forward(self, x, y):
        self.x, self.y = x, y          # kept for backward
        return x * y

    def backward(self, grad):          # grad is dL/dz for z = x * y
        return grad * self.y, grad * self.x
```

Calling `backward()` on the loss sorts the graph so every tensor comes after its inputs, then walks it from the loss back, handing each operation the gradient at its output. Condensed from stackgrad/tensor.py (the real one loops instead of recursing):

```python
def backward(self):
    topo, seen = [], set()
    def build(t):                        # every tensor after the tensors it was computed from
        if t not in seen:
            seen.add(t)
            if t._ctx:
                for parent in t._ctx.parents:
                    build(parent)
            topo.append(t)
    build(self)
    self.grad = ones_like(self.data)     # dL/dL = 1
    for t in reversed(topo):             # from the loss back to the inputs
        if t._ctx:
            for parent, g in zip(t._ctx.parents, t._ctx.backward(t.grad)):
                parent.grad = g if parent.grad is None else parent.grad + g
```

That loop is backprop. Everything else in a framework feeds it or makes it fast.

All of stackgrad's gradients live in mlops.py, in nineteen operations like add, multiply, exponential, logarithm, sum, max, reshape, broadcast and matrix multiplication. Subtraction adds a negation and division multiplies by a reciprocal. Softmax, normalization and the loss are compositions with no gradient of their own. I love that every gradient in the framework fits in one file.

Broadcasting is the one that bites. NumPy copies a number across an array, each copy gets a gradient on the way back, and stackgrad's Expand operation sums those gradients into the original.

### Checking the gradients

A wrong gradient doesn't crash anything. The model trains worse and nothing points you to the cause, so stackgrad checks its gradients two ways. The first is the definition of a derivative. Nudge an input up by a tiny $h$, then down, and measure the slope:

$$\frac{\partial f}{\partial x}\approx\frac{f(x+h)-f(x-h)}{2h},\qquad h=10^{-5}$$

It's slow, but it needs no calculus and no trust. stackgrad runs it on 52 operations and layers.

> **Figure:** Finite-difference gradient checks, replayed from stackgrad's server at the pace they ran. A line passes when `backward()` matches the measured slope for every input.

The second is PyTorch. I trained the same 8.3M-parameter language model in both for 200 steps, with the same starting weights, batches and optimizer. One wrong gradient anywhere and the loss curves would split.

> **Figure:** Training loss of the same model in stackgrad (thin blue) and PyTorch (wide pink), in float64, 1.8e-15 apart at step 1 and at most 8.4e-4 apart over 200 steps. Switch to float32 to compare.

They differ at all because a 64-bit float keeps about 16 significant digits, and adding numbers in a different order changes the last ones. The step-1 gap of 1.8e-15 is rounding alone, so the math is identical, and training compounds it to 8.4e-4. In 32-bit floats (about 7 digits, what models use in practice) each rounding is about half a billion times larger and the gap reaches 1.3e-2.

On the laptop's CPU a step takes 191 ms in stackgrad and 90.1 ms in PyTorch, so PyTorch runs 2.1 times faster. Not bad for plain Python.

### Laziness

So far every operation ran the moment you wrote it. Now stackgrad waits. `c = a @ b` records the multiplication and returns a tensor with a shape and dtype but no numbers. Reading it through `.numpy()`, `.item()`, printing, `.backward()` or `.realize()` runs the graph, and [tinygrad](https://github.com/tinygrad/tinygrad) and MLX work the same way.

```python
c = a @ b      # recorded: c.shape is known, nothing has run
y = layer(x)   # recorded and returned: still nothing has run
y.numpy()      # realize: every operation y depends on runs, inputs first, once each
```

An operation nobody reads never runs. Each of the 20 primitives in mlops.py (the nineteen plus the comparison `x < y`) states its output's shape and dtype up front, so the next operation can be recorded before the last one has run.

Waiting saves nothing by itself. A training step of a 2-layer story model with 128-wide vectors records 178 operations and runs all 178, since every result feeds the loss. On the CPU lazy matches eager bit for bit, at 10 ms a step against 10 ms. On the Apple GPU a step takes 4.8 ms against 4.09 ms, and lazy and eager losses differ by 4.8e-7, no more than two eager runs do, because the GPU sums in a different order each run. Laziness pays off once a compiler fuses the recorded graph into fewer kernels. stackgrad stops before that (bench/lazy.py, tests/lazy.py), so for now it's all setup.

## Tokenization

A language model reads integers, so a tokenizer cuts text into tokens from a fixed list, the vocabulary, and replaces each token with its position in that list.

Single characters make a tiny vocabulary but long sequences, and the model must learn to spell first. Whole words make short sequences, but the vocabulary explodes and a name or a typo has no token. Modern models use subwords, one token for a common word and a few for a rare one.

stackgrad uses byte-level byte-pair encoding ([BPE](https://arxiv.org/abs/1508.07909), as in [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)), the scheme behind GPT-2, GPT-4 and Qwen. It starts from the 256 possible bytes, so no UTF-8 text, emoji included, is ever unknown. Then it learns merges. Find the most frequent adjacent pair in the training text, add it to the vocabulary as a new symbol, and repeat until the vocabulary is the size you asked for.

Say the training text has `low` 5 times, `lower` 2 times, `newest` 6 times and `widest` 3 times. Weighted by word count, `e s` and `s t` tie at 9, so the first merge creates `es`. Recount, and `es t` leads with 9, so the second creates `est`. Then `l o` and `lo w` lead with 7 each. After four merges `low` is one token and `newest` is `n e w est`.

Encoding new text replays the merges in training order, after a regular expression splits it into words, numbers and punctuation so no merge spans two words. Training counts word frequencies in parallel and, after each merge, recounts pairs only in the words that held the merged pair.

A 10,000-token vocabulary trained on TinyStories, then used to encode the whole training set:

| Quantity | Value |
|---|---|
| Vocabulary size | 10,000 tokens |
| Training time | 44.7 s |
| Peak memory of the largest worker process | 547 MB |
| Tokens in the training set | 541,229,347 |
| Encoding speed | 90.1 MB/s |
| Average bytes per token | 4.116 |
| Decoding the held-out set back to text | byte-for-byte exact |

An average token covers 4.12 bytes, so the model reads about a quarter as many tokens as there are characters.

> **Figure:** The tokenizer on stackgrad's server. Pick a text to see its tokens and ids, then a word to watch its bytes merge.

With GPT-2's published merges, the tokenizer matches tiktoken, OpenAI's tokenizer library, id for id. With Qwen3's vocabulary, regular expression and Unicode normalization, it matches Hugging Face's on stories, Chinese, emoji, code and chat transcripts. Qwen's regular expression keeps each digit a separate token so numbers stay easy to do arithmetic on, a smart call.

Chat models also reserve special tokens that ordinary text never produces. `<|endoftext|>` ends a document, and `<|im_start|>` and `<|im_end|>` mark who is speaking. My hand-written chat template wraps messages in them and matches Hugging Face's character for character.

## The transformer

Given some tokens, a language model gives every token in the vocabulary a probability of coming next. Sample one, append it, ask again. Chat models write stories and do arithmetic this way.

stackgrad's model is a decoder-only [transformer](https://arxiv.org/abs/1706.03762), the architecture behind GPT, Llama and Qwen, in the modern form [Stanford's CS336](https://stanford-cs336.github.io/) teaches. Let's follow one token through it.

The token id picks a row of a learned table: $d$ numbers (512 for the story model), the model's working notes on that position. The notes flow up through identical layers, each adding its own update, and that running sum is the residual stream.

Attention lets a position gather information from earlier ones. Each position computes a query ("what am I looking for?"), a key ("what do I contain?") and a value ("what do I pass on?"). Query dot key gives a score, softmax turns scores into weights that sum to 1, and the output is the weighted sum of values:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}} + M\right)V$$

The mask $M$ holds $-\infty$ wherever a position would look ahead, so nothing sees the future it is predicting, and $\sqrt{d}$ keeps scores from growing with vector length. A layer runs several of these heads side by side. In [grouped-query attention](https://arxiv.org/abs/2305.13245), several query heads share one set of keys and values to save memory during generation.

Most of the trained story model's heads smear their weight around, but a few have habits. In layer 3, head 11, words inside a phrase look back to the verb or preposition that opens it: little, girl and the first Lily look at was, red and ball at with. Layer 1, head 15 parks much of its weight on the first token. Nobody programmed either habit. Training found them.

> **Figure:** Every head of the trained story model on one sentence. Bluer cells carry more weight (square-root scale).

Attention alone ignores word order. Shuffle the earlier tokens and the sum stays the same. [RoPE](https://arxiv.org/abs/2104.09864) rotates each query and key by an angle proportional to its position $p$, at a different speed for each pair of dimensions:

$$\begin{pmatrix} x_i \\ x_{i+d/2} \end{pmatrix} \mapsto \begin{pmatrix} \cos p\theta_i & -\sin p\theta_i \\ \sin p\theta_i & \cos p\theta_i \end{pmatrix} \begin{pmatrix} x_i \\ x_{i+d/2} \end{pmatrix}, \qquad \theta_i = b^{-2i/d}$$

Rotate both vectors and their dot product depends on the difference of their angles alone, so scores depend on the distance between tokens. It's a neat trick.

Then each position runs its notes alone through [SwiGLU](https://arxiv.org/abs/2002.05202), where one projection of the input, passed through a smooth on-off function (SiLU), gates a second. [RMSNorm](https://arxiv.org/abs/1910.07467) before each block divides the notes by their root mean square and multiplies by learned gains $g$, so values neither vanish nor explode:

$$\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot g, \qquad \mathrm{FFN}(x) = W_2\big(\mathrm{SiLU}(W_1 x) \odot W_3 x\big)$$

That's a whole layer, two lines in stackgrad/gpt.py:

```python
def forward(self, x, cache=None, pad=None):
    x = x + self.attn(self.attn_norm(x), cache, pad)   # attention, added to the residual stream
    return x + self.ffn(self.ffn_norm(x))              # feed-forward, added the same way
```

Stack that and you have GPT. At the top, a final normalization and one more matrix give every vocabulary entry a score, its logit, and softmax turns those into probabilities. Qwen3-0.6B, loaded later, is the same design with four settings changed, so one class covers both models:

| Setting | Story model (CS336) | Qwen3-0.6B |
|---|---|---|
| Vocabulary | 10,000 | 151,936 |
| Vector width $d$ | 512 | 1,024 |
| Layers | 4 | 28 |
| Query heads / key-value heads | 16 / 16 | 16 / 8 |
| Width of one head | 32 | 128 |
| Feed-forward width | 1,344 | 3,072 |
| RoPE base $b$ | 10,000 | 1,000,000 |
| Normalization on queries and keys | no | yes |
| Output matrix shared with the embedding | no | yes |
| Parameters | 22,696,448 | 596,049,920 |

## Training it

Training needs a loss. For next-token prediction it's cross-entropy, the negative log of the probability the model gave the token that came next. Probability 1 costs 0, and 0.25 costs $-\ln 0.25 \approx 1.39$. Random guessing among 10,000 tokens scores $\ln(\text{10,000}) \approx 9.21$, and the untrained model's first step scores 9.23, which makes a good first sanity check. The exponential of the loss, the perplexity, reads easier: 4 means the model is as unsure as a uniform pick among 4 tokens.

The data is [TinyStories](https://arxiv.org/abs/2305.07759), millions of short children's stories GPT-4 wrote in a small child's vocabulary. A small model learns to write coherent stories from it in an hour on one GPU, which makes it perfect for this. Training repeats one step 40,000 times on 32 random windows of 256 tokens (train_gpt.py, condensed):

```python
for step in range(1, steps + 1):
    opt.lr = cosine_lr(step, max_lr, min_lr, warmup, steps)   # warmup, then cosine decay
    x, y = get_batch(train, batch, context)                   # y is x shifted by one token
    loss = cross_entropy_loss(model(x), y)
    opt.zero_grad()
    loss.backward()
    clip_grad_norm(params, 1.0)
    opt.step()                                                # AdamW
```

AdamW ([Adam](https://arxiv.org/abs/1412.6980) with [decoupled weight decay](https://arxiv.org/abs/1711.05101)) gives each weight its own step size from running averages of its gradient and squared gradient, so weights with small gradients still move and noisy ones move less, and it pulls every weight a little toward zero. The learning rate warms up over 1,000 steps, then follows a cosine down to 0.0001. Clipping rescales any gradient longer than 1, so one odd batch can't wreck the weights. Every 500 steps we score held-out stories to catch memorizing.

The full run used CS336's configuration: the 22.7M-parameter story model, 327.7M tokens, learning rate 0.001, weight decay 0.1, on one NVIDIA GeForce RTX 3090 through stackgrad's CuPy backend.

> **Figure:** The TinyStories run (runs/tinystories), train loss in blue and validation in pink, log scale. It ends at a validation loss of 1.399, a perplexity of 4.05.

Validation loss falls from 2.807 after the first 500 steps to 1.399, under CS336's target of 1.45, in 71 minutes at a median 80,643 tokens per second. The two curves track each other the whole way, so the model handles unseen stories about as well as its training ones. A sample from the final checkpoint, cut off at its length limit:

```text
Once upon a time, there was a big dog named Max. Max liked to chew on his toys all day long. He lived in a small house with his friend, a little girl named Lily.

One day, Max and Lily were playing in the yard. Max found a shiny thing on the ground. He picked it up with his mouth and showed it to Lily. "Look, Lily! I found a shiny thing!" Max wagged his tail and barked.

Lily looked at the shiny thing and said, "
```

Names, dialogue, a dog wagging its tail, all trained through the autograd engine from the start of this post. That's what all the autograd plumbing was for.

When sampling, temperature divides the logits before the softmax: below 1 sharpens toward the likeliest token, above 1 flattens, and 0 always takes the likeliest. Top-p keeps the smallest set of likeliest tokens whose probabilities add up to p.

> **Figure:** Stories from the trained checkpoint at three temperatures (top-p 0.95, at most 120 new tokens).

### Which parts matter

Do the modern parts (RMSNorm before each sublayer, RoPE, SwiGLU) matter? CS336's ablations take each one out and retrain the story model for 10,000 steps, a quarter of the full run (ablations.json).

As built, it ends at a validation loss of 1.524. Moving RMSNorm after each sublayer, where the original transformer put it, costs 0.013. Swapping SwiGLU for a plain SiLU layer of the same parameter count costs 0.03. Dropping position information entirely costs 0.085, and the model still learns! [Haviv et al.](https://arxiv.org/abs/2203.16634) traced this to the causal mask. Each position sees a different number of earlier tokens, which tells it roughly where it is. I find that wild.

Without RMSNorm nothing rescales the vectors between sublayers, and at the full learning rate of 0.001 the loss stopped being a number at step 627, 127 steps after warmup reached the peak rate. At 0.0003 it trains to 1.699, 0.18 behind the model with RMSNorm at its own rate. So the other three are polish, and normalization is what keeps the run alive.

One more switch, `TransformerLM(..., attn_res=True)`, swaps the plain residual sum for Block Attention Residuals: a learned softmax before each sublayer mixes earlier layers' outputs. I haven't trained the story model with it. My Attention Residuals post measures it on a small model.

## GPUs

Most of a model's arithmetic is one operation over huge arrays. A CPU has a handful of fast cores that run complicated programs step by step. A GPU has thousands of simple cores running one small program, a kernel, on different pieces of an array at once.

stackgrad runs on three array libraries with NumPy's interface: NumPy on the CPU, [MLX](https://github.com/ml-explore/mlx) on the Apple laptop's GPU, and [CuPy](https://github.com/cupy/cupy) on an NVIDIA GPU. Each op in mlops.py gets a module, `xp`, with the same dozen functions, so one copy of every gradient runs everywhere. That one interface is what keeps three backends cheap. Of 52 gradient checks against NumPy, 52 agree on the Apple GPU and 51 on the NVIDIA GPU, and the story model's loss stays within 4.8e-7 of NumPy's.

### A GPU does not win by default

Every kernel launch has a fixed cost (the CPU describes the work, the GPU schedules it), and for a small array that's the whole cost. Multiplying a thousand numbers takes 3.1 µs with NumPy, 7.6 µs on the Apple GPU and 42 µs on the NVIDIA GPU. On ten million it's 863 µs, 778 µs and 99.7 µs, and the NVIDIA GPU wins by 8.7x.

> **Figure:** One elementwise multiply's cost against array size. Waiting for the GPU after every op exposes each round trip: 139 µs on the Apple GPU, 35 µs on the NVIDIA GPU.

So a GPU pays off only when each kernel gets enough work to bury its launch cost.

Matrix multiplication gets plenty. Two $n \times n$ matrices take $2n^3$ floating-point operations (FLOPs) on $3n^2$ numbers, so as $n$ grows the arithmetic outgrows the memory traffic. The NVIDIA GPU peaks at 33.9 TFLOP/s (trillions of operations per second) using TF32, a reduced-precision format its tensor cores run in hardware, and the Apple GPU at 3.62 TFLOP/s.

> **Figure:** Square matmul throughput against matrix size. Dashed lines use bfloat16 on the Apple GPU and TF32 on the NVIDIA GPU.

[Mixed precision](https://arxiv.org/abs/1710.03740), multiplying in 16 bits while the weights stay in float32, takes a story-model step on the NVIDIA GPU from 69.6 ms to 61.8 ms and moves the loss by 0.0073.

A training step is mostly large matmuls, and there the NVIDIA GPU trains 25.1x faster than the laptop's CPU. On the Apple GPU, stackgrad reaches 82% of the speed of the same model in plain MLX with its hand-tuned kernels. I call that a fair price for generality.

> **Figure:** One training step of the story model on each device.

Model FLOPs utilization (MFU) is the arithmetic a step must do, about six operations per parameter per token, over the chip's rated peak. The full TinyStories run sustained a median 8.5 TFLOP/s, an MFU of 23.9%. Launch overhead and ops that move memory without much arithmetic eat the rest. Attention is full of those, so next we fuse it into one kernel.

## Writing an attention kernel

Back in the transformer part, attention was four steps: scores $S = QK^\top/\sqrt{d}$ for every pair of positions, a mask, a softmax, and a weighted average of the values. In stackgrad each is its own kernel, and for 256 tokens and 16 heads each one writes a million scores per sequence to GPU memory for the next to read back.

A GPU does arithmetic faster than it moves data, so that traffic is the real cost. [FlashAttention](https://arxiv.org/abs/2205.14135) does all of attention in one kernel that never writes the scores, using a trick I think everyone should know, [the softmax in one pass](https://arxiv.org/abs/1805.02867).

A softmax needs the largest score $m$ and the sum $l = \sum_j e^{s_j - m}$, which seem to need every score up front. Instead, keep a running max and sum, and when a bigger score arrives, rescale the sum so far by $e^{m_\text{old} - m_\text{new}}$. Take the scores 1, 3, 2. After the 1, $m = 1$ and $l = 1$. The 3 raises the max, which shrinks the old sum to $e^{1-3} = 0.135$, and adds $e^0 = 1$, for $l = 1.135$. The 2 adds $e^{-1} = 0.368$, for $l = 1.503$, the same as computing it all at once. Rescale the weighted values the same way and one pass gives the output:

$$
\begin{aligned} m_j &= \max(m_{j-1},\, s_j) \\ l_j &= l_{j-1}\,e^{m_{j-1}-m_j} + e^{s_j-m_j} \\ o_j &= o_{j-1}\,e^{m_{j-1}-m_j} + e^{s_j-m_j}\,v_j \end{aligned} \qquad \text{output} = \frac{o_S}{l_S}
$$

I wrote three Metal kernels for the Apple GPU (with MLX's `mx.fast.metal_kernel`):

- v1 runs a thread per query position, and neighbouring threads re-read the same keys.
- v2 has 32 query positions share a tile of keys and values in on-chip memory, so each key leaves main memory 32 times less often.
- v3 keeps the tiles and multiplies 8 by 8 matrices in hardware with simdgroup matrix instructions.

15 of 15 runs match a 64-bit reference to within 1e-4. Times in milliseconds ("materialized scores" is the version built from basic ops), against MLX's `scaled_dot_product_attention`, which has a fused kernel for heads 64 or 128 wide but not 32:

| Shape | Format | materialized scores | v1 thread per row | v2 threadgroup tiles | v3 simdgroup matmul | mx.fast.sdpa (oracle) |
|---|---|---|---|---|---|---|
| CS336 training | float32 | 9.94 | 3.02 | 2.91 | **1.49** | 9.94 |
| CS336 training | bfloat16 | 11.2 | 2.03 | 3.49 | **1.65** | 5.4 |
| prefill 1k | float32 | 20 | 32.3 | 40.5 | 6.22 | **3.31** |
| prefill 1k | bfloat16 | 22.2 | 28.2 | 43.2 | 7.55 | **2.74** |
| Qwen3 prefill 2k | float32 | 26.3 | 329 | 124 | 18.8 | **7.19** |
| Qwen3 prefill 2k | bfloat16 | 26.8 | 303 | 127 | 19.8 | **5.6** |
| decode, 2k cache | float32 | 7.63 | 1.55 | 36 | 5.34 | **1.39** |
| decode, 2k cache | bfloat16 | 6.91 | 1.02 | 33.9 | 6.01 | **0.712** |

At the training shape (32-wide heads) v3 runs 6.7x faster than MLX. At Qwen3's 128-wide heads it is 2.6x slower than MLX's fused kernel. So I win where the library has no fused kernel and lose where it has one. I'll take that.

Training also needs a backward pass. The kernel returns the log of each row's softmax sum, enough to rebuild $P$ in backward instead of storing it ($dO$ is the gradient arriving at the output):

$$
\begin{aligned} P &= e^{S - \text{lse}} & dV &= P^\top dO & dP &= dO\,V^\top \\ dS &= P \odot \big(dP - \operatorname{rowsum}(dO \odot O)\big) & dQ &= dS\,K / \sqrt{d} & dK &= dS^\top Q / \sqrt{d} \end{aligned}
$$

### Where the time goes

A kernel is worth writing only where the time goes, so ask a profiler. Inside `with profile():` each primitive waits for the GPU before the next one starts, so each time is its own. A profiled forward and backward pass of the story model takes 777 ms on the Apple GPU (`profile_metal.json`).

After matrix multiplies, attention costs the most, and nearly all of it is backward: 121 ms against 8.3 ms for the fused forward. That's 14.5x, where counting arithmetic predicts 4.4x. The composed backward in `mlops.py` writes $P$, $dP$ and $dS$, three score-sized matrices, to GPU memory and reads them back, the traffic the forward kernel exists to avoid.

### The backward in Triton

On the NVIDIA GPU I wrote [FlashAttention-2](https://arxiv.org/abs/2307.08691)'s forward and backward in [Triton](https://github.com/triton-lang/triton), a Python dialect where you describe one tile's work and the compiler handles the threads. The backward rebuilds each tile of $P$ on-chip from the saved log-sum-exp, so no score-sized matrix reaches main memory. 5 of 5 checks match a 64-bit reference to within 1e-4.

At the training shape, forward plus backward takes 0.764 ms against 6.48 ms composed, 8.5x faster (`flash_attention_cuda.json`). Attention's backward takes 3.8% of a profiled NVIDIA training step, where the composed one took 15.6% on the Apple GPU (`profile_cuda.json`).

Profile before you write a kernel. The forward is the famous half of FlashAttention, and the profile puts the time in the backward.

## Running someone else's model

A stronger test than my own model is someone else's, loaded from its published weights and checked against the reference library. [Qwen3-0.6B](https://arxiv.org/abs/2505.09388) is a small open chat model from Alibaba's Qwen team that can hold a conversation and do grade-school math.

The weights ship as safetensors: eight bytes of header length, a JSON header with each tensor's name, number type, shape and byte range, then raw bytes. Loading runs no code from the file, which is the whole point.

The bytes are bfloat16, the first half of a float32. It keeps the sign and the exponent (the scale) but only 7 of the 23 mantissa bits (the digits), which leaves two to three significant digits instead of seven, over the same range, in half the memory. Converting to float32 appends 16 zero bits and loses nothing.

Qwen3 is the transformer from earlier with four settings changed: 128-wide heads, queries and keys normalized before RoPE, a smaller normalization constant, and an output layer that reuses the embedding table. The loader refuses to run on any mismatched tensor name or shape. Every loader should do this. A weight left at its random starting value raises no error, and the model talks nonsense.

stackgrad's logits on a 17-token prompt against Hugging Face's transformers (Qwen3's largest logit is 24.3):

| Model | stackgrad on | Largest gap | Gap relative to the largest logit | Same next token |
|---|---|---|---|---|
| Qwen3-0.6B | CPU (NumPy), float32 | 5.1e-5 | 2.1e-6 | 100% |
| Qwen3-0.6B | Apple GPU, float32 | 9.7e-5 | 4.0e-6 | 100% |
| Qwen3-0.6B | Apple GPU, bfloat16 | 3.5e-1 | 1.4e-2 | 100% |

In float32 the gap is a few millionths of the logits' size, which is rounding. In bfloat16, the format people run the model in, it grows to 1.4% of the largest logit, and both still pick the same next token at every position. That last column is the one I care about. Greedy decoding reproduces Hugging Face's first 24 tokens.

The same layers also run [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), from one tinygrad-style file in the bonus section.

## Serving more than one person

A chat reply takes one forward pass per new token. Without a cache, step $t$ recomputes the keys and values of all $t$ earlier tokens, though they never change. The KV cache stores them, so each step computes keys and values for the new token alone. You pay in memory, since every token of every conversation keeps its keys and values for every layer.

That memory limits how many people one GPU can serve. Instead of reserving the maximum length for each request, vLLM's [PagedAttention](https://arxiv.org/abs/2309.06180) hands out the cache in small fixed-size blocks as a sequence grows. stackgrad's block manager keeps the same books, but the memory stays one array per cache. I took the bookkeeping and skipped the paging.

The other idea is [continuous batching](https://www.usenix.org/conference/osdi22/presentation/yu). Instead of holding a batch until its longest reply finishes, the scheduler admits new requests between any two steps and lets each one leave when it's done. When the blocks run out, it preempts the newest request, which later rebuilds its cache from its prompt and the tokens it already produced.

To batch, the runner left-pads every cache to the longest, masks the padding, and keeps each token at its own position for RoPE. When a request joins or leaves, the runner re-pads and copies every cache, and the padding is memory the block manager never counts. I like that part of the design least.

`tests/serve.py` checks that under greedy decoding each request gets the same tokens it gets running alone. It does under all 3 loads, including 3 preemptions in a cache of only 9 blocks (`serve_cpu.json`).

> **Figure:** Qwen3-0.6B served by stackgrad on the Apple GPU in bfloat16. Pick a message and send it. The reply is a recording, replayed at the speed it streamed, with time to first token and speed underneath.

The key result: a step for several conversations costs little more than one. A decode step pays to read every weight and to launch hundreds of small kernels, and neither cost grows with the number of rows. With 8 chats arriving at once, each writing 128 tokens, one pass per conversation produces 21.4 tokens per second in all, about what one chat gets alone (22.6). One pass per step produces 128, 6x as many, while the step grows from 44.4 ms to 58.3 ms.

In float32 all 8 batched replies match the one-at-a-time ones token for token. In bfloat16, 4 of 8 do, since a masked batch skips the fused attention kernel and rounds differently, and one flipped near-tie changes the rest of a reply (`serve_batch_metal.json`). Each reply still streams no faster than one decode step allows.

## Making decoding faster

Each reply in the chat above streams at one token per decode step, and each step reads every weight from memory once, about 1.2 GB for a 0.6B-parameter model in bfloat16. Arithmetic is cheap next to that, so memory bandwidth sets the speed. You can read fewer bytes per token or get more tokens per read.

Quantization reads fewer bytes by rounding each weight to one of $2^{\text{bits}}$ levels, with a scale and offset per group of 64 (`tests/quant.py`). Four bits cut the weights read per token from 1,192 MB to 335 MB, for 19.3% higher perplexity on math text, yet the step goes from 24.7 ms to 20.9 ms. In bfloat16, reading the weights takes 11.9 ms of the step, and the rest is small operations that each pay a kernel launch (`quant.json`).

Speculative decoding ([Leviathan et al.](https://arxiv.org/abs/2211.17192), [Chen et al.](https://arxiv.org/abs/2302.01318)) gets more tokens per read. A small draft model guesses a few tokens, and the big model checks them all in one pass without changing its output distribution (`tests/speculative.py`). With Qwen3-0.6B drafting for Qwen3-1.7B, the target kept 80% of single guesses, and the best speedup was 1.07x (`speculative_metal.json`). Both models have 28 layers and launch the same number of kernels, so the draft's step takes 61% as long as the target's.

Both ideas work, and neither buys much here, because neither removes the kernel launches.

## Fine-tuning with LoRA

Fine-tuning trains a pretrained model a bit further on examples of the behaviour you want. Updating every weight is expensive: Adam keeps two extra numbers per weight, so a 0.6B model needs memory for several copies of itself.

[LoRA](https://arxiv.org/abs/2106.09685) freezes the original weights and learns a small correction $AB$ next to each matrix $W$, where $A$ has $r$ columns and $B$ has $r$ rows for a small rank $r$:

$$y = xW + \frac{\alpha}{r}\,(xA)\,B, \qquad \underbrace{n_\text{in} \cdot n_\text{out}}_{\text{frozen}} \;\gg\; \underbrace{r\,(n_\text{in} + n_\text{out})}_{\text{trained}}$$

$B$ starts at zero, so training starts from the unchanged model. At rank 16, a 1024 by 1024 matrix gets 32,768 trainable numbers against its own 1,048,576. Across Qwen3-0.6B that's 10,092,544 numbers, 1.69% of the model. After training you add $AB$ into $W$ once, and the fine-tuned model runs at the original's speed. No extra cost at inference, ever.

The task is [GSM8K](https://arxiv.org/abs/2110.14168), grade-school word problems with worked solutions ending in `#### answer`, and the model trains on the 7,473 training problems. I fixed the evaluation before training: the first 250 test problems, greedy decoding, at most 320 new tokens, solved when the final number equals the reference. Here are the results on the Apple GPU in bfloat16, where "hit the limit" counts replies cut off at 320 tokens.

| Model | Solved | Accuracy | Average reply | Hit the limit |
|---|---|---|---|---|
| Qwen3-0.6B as released | 152 of 250 | 60.8% | 213 tokens | 45 |
| after LoRA on GSM8K | 123 of 250 | 49.2% | 92 tokens | 11 |

LoRA made the model *worse* at math, 60.8% down to 49.2%, while the training loss on the solution tokens fell from an average of 0.770 over the first 50 steps to 0.573 over the last 50. It got better at reproducing GSM8K's solutions, which are terse, and it learned their length along with their format, from 213 tokens per reply down to 92. The released model writes out its steps, and those steps were getting answers right.

> **Figure:** The LoRA run's training loss on the solution tokens only, from `runs/lora_gsm8k`. The eval in the table above is the validation curve here.

Problem 56 shows the change. Peter buys a $7 ticket and $7 popcorn each time, with $42 to spend. The released model, in 83 tokens:

```text
Step 1: Peter pays $7 for a movie ticket and $7 for popcorn.
So, each movie visit costs **$14**.

Step 2: He has **$42** for the week.
How many times can he go to the movies?

$$
\frac{42}{14} = 3
$$

**Final Answer:**
#### 3
```

After LoRA, 28 tokens in GSM8K's house style, and the popcorn is gone:

```text
Peter can buy 42/7 = 6 tickets.
So he can go to the movies 6 times.
#### 6
```

The lesson I'd take away: the loss is a proxy, so measure the thing you care about.

One bug worth telling. My first run trained in float64 by mistake. Each adapter's $A$ starts as random numbers divided by $\sqrt{n_\text{in}}$, and since NumPy 2, dividing a float32 array by NumPy's float64 square root gives float64. The first adapted layer's output went float64, and every layer after it followed, forward and backward, on a 3090 that does float64 at 1/64 of its float32 rate. That run took 136 minutes against 14.8 minutes with a cast, and its adapter solved 124 of 250 to the float32 adapter's 123. `tests/lora.py` now checks that the adapter and the loss stay float32.

## Learning from its own answers

LoRA trained the model to copy solutions, and it got worse at the thing the table counts. So let's train on the count itself. In reinforcement learning the model writes its own replies, a checker scores each one (1 if the final number is right, 0 if not), and training makes the high scorers likelier. Nobody tells the model how to reason.

The simplest version is the [policy gradient](https://link.springer.com/article/10.1007/BF00992696). Sample a reply, then nudge the weights to make it likelier in proportion to its score. That pushes every correct reply up by the same amount, even on a question the model never gets wrong. [GRPO](https://arxiv.org/abs/2402.03300) samples a group of replies to each question and scores each one against its siblings:

$$A_i = \frac{R(o_i) - \operatorname{mean}_j R(o_j)}{\operatorname{std}_j R(o_j) + \epsilon}, \qquad \mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} A_i\, \frac{1}{B} \sum_t \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})$$

$R$ is a reply's score and $\pi_\theta$ the model's probability of each token. Say one reply of four is right, scores 1, 0, 0, 0. The mean is 0.25 and the standard deviation 0.433, so the right reply gets advantage 1.73 and each wrong one gets $-0.58$. The right reply gets pushed up three times as hard as each wrong one gets pushed down, so the pushes cancel: $1.73 = 3 \times 0.58$. A group that scored the same everywhere gets all-zero advantages, and stackgrad skips it. $B$ is the 320-token budget, the same for every reply, because dividing by a reply's own length would give a long wrong reply a smaller push per token ([Liu et al.](https://arxiv.org/abs/2503.20783)). In code, condensed from `policy_loss` in `train_grpo.py`:

```python
logp = log_softmax(logits, axis=-1)[rows, targets]                  # each reply token's log-probability
weight = np.repeat(advantages / budget / n_replies, reply_lengths)  # A_i / (B N), once per token
loss = -(logp * weight).sum()
```

Training starts from the released Qwen3-0.6B with a fresh LoRA adapter. Each step samples 8 replies to each of 8 training problems at temperature 1 and makes one update, and a step averages 39 s of sampling and 7.3 s of updating. As we saw back in the GPU part, one reply leaves most of the GPU idle, so all 64 go in one batch: on the 3090 a decode step takes 79 ms for one reply and 86.4 ms for 64. The first version still crawled, because it copied all 64 rows of logits, 151,936 numbers each, to the CPU to draw the tokens. That took 98.5 ms per step against the model's 84.3 ms; on the GPU it takes 1.2 ms.

### The reward the model gamed

The first run rewarded the final number alone. From step 51 on, every reply it sampled ran to the 320-token limit: the model wrote its solution, then "The final answer is $\boxed{18}$." between rows of dashes, over and over. The number stayed right, and writing on cost nothing. That adapter solves 154 of 250 test problems against 152 before training, with all 250 replies hitting the limit. This is my favorite result in the project. The model optimized the reward I wrote, loophole included.

The fix makes stopping part of the answer: a reply earns 1 only if it ends with Qwen's end-of-turn token *and* its final number is right.

> **Figure:** Replies cut off at the 320-token limit, per step, under each reward (`runs/grpo_gsm8k`, `runs/grpo_no_stop`). The first run stopped at step 143.

With the new reward the cut-off share fell from 27% over the first ten steps to 2% at its lowest ten-step stretch, and ended at 8%. The reward on the training problems rose from 54% over the first 20 steps to 67% over the last 20.

> **Figure:** Sampled replies that scored on the training problems, per step, at temperature 1 (`runs/grpo_gsm8k`, 200 of 200 steps).

The test set, under the same protocol as the LoRA eval:

| Model | Solved | Accuracy | Average reply | Hit the limit |
|---|---|---|---|---|
| Qwen3-0.6B as released | 152 of 250 | 60.8% | 213 tokens | 45 |
| after LoRA on GSM8K | 123 of 250 | 49.2% | 92 tokens | 11 |
| after GRPO, reward ignores stopping | 154 of 250 | 61.6% | 320 tokens | 250 |
| after GRPO, reward requires stopping | 139 of 250 | 55.6% | 216 tokens | 29 |

On problem 56 the adapter writes 183 tokens to the released model's 83, and drops the popcorn the same way the LoRA model did (shortened):

```text
Each movie ticket costs $7 and popcorn also costs $7.
...
Each movie costs $7$ dollars.
...
So, $ 7n \leq 42 $

Therefore, $ n \leq \frac{42}{7} = 6 $
...
**Answer:** $\boxed{6}$
```

Is 139 against 152 a real drop? The adapter fixed 25 problems and broke 38. If training had changed nothing, each of those 63 would be a coin toss, and a split at least this uneven comes up 13% of the time, so I can't call it one. LoRA's 23 fixed against 52 broken comes up 0.1% of the time, so that drop was real. I find this ending disappointing: the training reward went up and the model learned to stop, but the test score stayed inside the noise.

## Retrieval and a calculator

Most systems built on language models answer from documents or call tools. Both run here as programs talking to the server from the serving part, with stackgrad computing every number from the embedding to the last sampled token.

### Retrieval

A small model knows few facts. [Retrieval-augmented generation](https://arxiv.org/abs/2005.11401) looks up passages that bear on the question and puts them in the prompt. The test is [SQuAD v1.1](https://arxiv.org/abs/1606.05250), questions people wrote about Wikipedia paragraphs; its development set has 2,067 paragraphs.

Finding the right paragraph means comparing meanings. An embedding model maps a text to a vector so that texts about the same thing point the same way. [Qwen3-Embedding-0.6B](https://arxiv.org/abs/2506.05176) is Qwen3's architecture with its own weights, so stackgrad already runs it, and a text's embedding is the last token's final hidden state scaled to length 1. The index is a matrix with one row per paragraph, a question takes 61 ms to embed, and the search is a matrix product and a sort:

```python
query = embed(model, tok, [INSTRUCT + question], device)[0]   # 1,024 numbers, length 1
scores = index @ query                                          # one dot product per paragraph
top = np.argsort(-scores)[:3]                                   # the three nearest
```

That's the whole search engine. For 200 random questions, the paragraph the question was written from ranks first 73.0% of the time and in the top 3 91.5% of the time, and the server answers with those top 3 in its prompt.

SQuAD scores answers by exact match and word overlap (F1). Qwen3 answers in sentences ("The chloroplast double membrane is sometimes compared to the mitochondrial double membrane."), which is right and scores 0 on exact match, so I also count answers that contain a reference answer as whole words. Asking without the paragraphs sets a floor, and since SQuAD's questions presuppose their paragraph, it's low. From `apps/rag.py`, on 200 questions:

| Prompt | Exact match | F1 | Contains | Seconds per answer |
|---|---|---|---|---|
| the question alone (a floor) | 0.0% | 0.085 | 3% | 0.48 |
| with the top 3 paragraphs | 20.5% | 0.4 | 55% | 1.53 |

Retrieval takes the answers that contain the reference from 3% to 55%, at 1.53 s per answer against 0.48 s. I'd pay that every time.

> **Figure:** Retrieval, recorded through stackgrad's server. stackgrad compares your question with all 2,067 development paragraphs, and Qwen3-0.6B answers from the three nearest.

### A calculator

GSM8K's problems take a few steps of arithmetic, and a 0.6B model slips on arithmetic, so give it a calculator. Qwen3 was trained to call tools: shown a function's description in its system prompt, it can stop mid-reply and write a JSON call between `<tool_call>` tags. `apps/agent.py` runs the [ReAct](https://arxiv.org/abs/2210.03629) loop. When a reply calls the tool, the program evaluates the expression and sends the value back, and the model continues until it answers without a call.

The first run kept the LoRA part's "Solve it step by step" prompt and offered the calculator in the system message. In 250 problems the model never called it, yet the offer changed the answers. Replies shrank from 213 tokens to 148, and the model solved 132 against 152, fixing 21 and breaking 41, a split chance produces 1.5% of the time. Told to solve step by step, it does.

So the second run asks for the calculator in the user's turn. The model called it 440 times, on 248 of 250 problems, and solved 42 against 152 without it. It fixed 7 and broke 117, a split chance produces less than 0.01% of the time. Replies averaged 27 tokens, little more than the calls. In the traces (`bench/results/gsm8k_agent_replies.jsonl`) the model hands the calculator its arithmetic before working out what to compute, then answers with whatever number comes back.

With the plain prompt a reply averaged 213 tokens of worked steps, and those steps were getting problems right. Asking for the tool pushed them out.

> **Figure:** The agent's loop, recorded on stackgrad's server, each reply and calculator call in order. It uses the second run's prompt, so expect the calls before the reasoning.

## Bonus: scaling laws

Was the story model the right size for its compute? At about six FLOPs per parameter per token, the full run spent $3.5 \times 10^{16}$, which could buy a bigger model on fewer tokens or a smaller one on more. [Hoffmann et al.](https://arxiv.org/abs/2203.15556) find the best split with IsoFLOP curves: fix a budget $C$, train several sizes $N$ on $C/6N$ tokens each, and see which ends lowest. ($N$ counts matrix weights only, 17.6M for the story model.) I ran three budgets, from $3 \times 10^{14}$ to $3 \times 10^{15}$, five sizes each.

> **Figure:** IsoFLOP curves from scripts/scaling.py, darker for more compute, with a ring at the bottom of each fitted parabola.

Too small a model can't hold what the data teaches, and too large a one sees too few tokens. The bottoms sit at 3M, 5.5M and 8.2M parameters, growing as $C^{0.43}$. I don't trust that exponent much, since the smallest budget's bottom sits at the edge of the grid, and the two larger budgets alone give $C^{0.37}$. At the full run's compute, 12 times past the largest budget, the fit asks for 24M parameters on 237M tokens (20M with the two-budget exponent). The run used 17.6M on 328M: close, and a little small.

## Bonus: GPT-2 and Stable Diffusion

Two more models run on stackgrad's layers from published weights, one file each, like tinygrad's examples/. GPT-2 (124M parameters) against transformers, on a 15-token prompt and 40 greedy tokens:

| stackgrad on | Largest gap | Relative to the largest logit | Same next token | Greedy text |
|---|---|---|---|---|
| CPU (NumPy) | 1.4e-4 | 7.0e-7 | 100% | identical |
| Apple GPU | 2.2e-4 | 1.1e-6 | 100% | identical |

[Stable Diffusion v1.5](https://arxiv.org/abs/2112.10752) turns a sentence into a 512 by 512 image with [CLIP](https://arxiv.org/abs/2103.00020)'s text encoder, an 860M-parameter UNet that denoises a 4 by 64 by 64 latent over [DDIM](https://arxiv.org/abs/2010.02502) steps with [classifier-free guidance](https://arxiv.org/abs/2207.12598), and a VAE decoder. examples/stable_diffusion.py builds all three:

```bash
uv run python examples/stable_diffusion.py --prompt "a lighthouse at dusk"
```

```python
# condensed from generate() in examples/stable_diffusion.py
tok = ClipTokenizer(path / "tokenizer")
prompts = np.array([tok.encode(""), tok.encode(prompt)])
context = load_clip(path, device)(prompts)          # 77 vectors per prompt
unet = load_unet(path, device)
for t, t_prev in zip(times, [*times[1:], -1]):     # 25 DDIM steps
    pair = Tensor(np.concatenate([x, x]), device=device)
    noise = unet(pair, t, context).numpy()
    noise = noise[:1] + guidance * (noise[1:] - noise[:1])   # guidance
    x = ddim_step(x, noise, t, t_prev)
image = load_vae(path, device).decode(Tensor(x, device=device))  # pixels
```

Each network against transformers and diffusers, largest difference over the reference's largest value (on smaller test latents):

| Network | Parameters | CPU (NumPy) | Apple GPU (MLX) |
|---|---|---|---|
| CLIP text encoder | 123M | 5.6e-7 | 6.9e-7 |
| UNet | 860M | 8.4e-6 | 4.5e-6 |
| VAE decoder | 49.5M | 3.0e-6 | 3.4e-6 |

> **Figure:** "a watercolor painting of a lighthouse on a cliff at dusk", 25 DDIM steps, guidance 7.5, same noise, Apple GPU. Pixels differ by at most 3.4 of 255 levels, and a step takes 2.58 s in stackgrad, 1.16 s in diffusers.

Slower, but the same picture.

## The code

Everything is at [github.com/samitmohan/minitorch](https://github.com/samitmohan/minitorch). Every number in this post comes from a script that writes it to a file there, and the README lists [the command behind each one](https://github.com/samitmohan/minitorch#every-number-in-the-write-up), so you can check me.
