---
gist: A GPU step waits on whichever is slower, the arithmetic or reading memory, and small models often wait on neither but on launching work, so you measure first, then reach for torch.compile and fused attention.
minutes: 17
---

## What this lesson is for {#why}
> Before you make generation faster, find out what it's actually waiting on. Speeding up the wrong thing changes nothing.

You've built a correct, cached, batched generator. This lesson is about speed, done properly:

1. **Arithmetic first.** For a decode step, work out how long the math takes and how long reading the weights takes. Whichever is slower sets the time. This is the **roofline** idea.
2. **Check before you swap.** PyTorch's `F.scaled_dot_product_attention` (SDPA) can run a fused FlashAttention kernel on the GPU. Confirm it computes exactly the attention you wrote in 1.1.
3. **Measure on the GPU.** Profile your 30M model, try SDPA and `torch.compile`, and record the speedup while checking the outputs haven't changed.

The same reasoning explains the rest of the course. Quantization (phase 3) speeds up decoding because it reads fewer bytes. Serving systems (phase 5) win by batching.

## What you need from before {#before}
> A matrix multiply does one multiply and one add per weight per token. fp16 numbers take 2 bytes. Batching (2.4) shares one read of the weights across every row.

- **Parameters.** Llama 3.1 8B has about 8 × 10⁹ weights. In fp16, at 2 bytes each, that's 16 GB.
- **Per token, per weight:** a multiply and an add, which counts as **2 FLOPs** (floating-point operations).
- **Decode** (2.2) runs one new token per sequence per step, using the KV cache.
- **Batching** (2.4) runs B sequences per step for nearly the same cost.
- **Attention** (1.1): scores = q · kᵀ ÷ √D, mask the future, softmax over the keys, mix the values.

## Two speeds of a GPU {#two-speeds}
> A GPU has a top speed for arithmetic (FLOPs per second) and a top speed for reading memory (bytes per second). A step can't beat either.

This lesson uses the RTX 3090's numbers:

| Limit | RTX 3090 |
|---|---|
| Peak fp16 compute (tensor cores, fp32 accumulate) | 71 TFLOPS = 71 × 10¹² FLOPs/s |
| Memory bandwidth | 936 GB/s = 936 × 10⁹ bytes/s |

A web analogy: a request handler is either CPU-bound (busy computing) or I/O-bound (waiting on the database or network). Making the code faster only helps the CPU-bound one. A GPU kernel is the same. It's either **compute-bound** or **memory-bound**, and you need to know which before you optimize.

## Counting one decode step {#count}
> Compute time is the FLOPs divided by the peak rate. Memory time is the bytes moved divided by the bandwidth. They overlap, so the step takes the larger of the two.

Llama 3.1 8B in fp16, batch 1, on the 3090:

| | Amount | Rate | Time |
|---|---|---|---|
| Math | 2 × 8×10⁹ = 16 GFLOP | 71 TFLOPS | 0.23 ms |
| Reading weights | 8×10⁹ × 2 = 16 GB | 936 GB/s | 17.1 ms |

The step takes **17.1 ms**, and it's **memory-bound**: the arithmetic units sit idle about 99% of the time, waiting for weights to arrive. A faster matmul would change nothing. Reading fewer bytes would.

Why the larger and not the sum? The GPU loads the next block of data while it computes on the current one, so the two overlap. The step finishes when the slower one finishes. In the challenge, adding the two times, or taking the smaller one, are the mistakes the tests look for.

Two other details the tests check:

- **FLOPs grow with the batch.** Every token in the batch does its own 2 FLOPs per weight.
- **Weight bytes don't.** The weights are read once per step and shared by every row, which is the whole point of 2.4. Also, `bytes_per_param` is already in bytes (2 for fp16), so there's no factor of 8.

## Batching and the ridge point {#ridge}
> Batching raises the work done per byte read. Past a certain batch size, compute becomes the slower side and the step starts to slow down.

Same model and GPU, growing batch:

| Batch | Compute | Memory | Step | Tokens/s | Bound |
|---|---|---|---|---|---|
| 1 | 0.23 ms | 17.1 ms | 17.1 ms | 58 | memory |
| 16 | 3.6 ms | 17.1 ms | 17.1 ms | 936 | memory |
| 64 | 14.4 ms | 17.1 ms | 17.1 ms | 3,744 | memory |
| 256 | 57.7 ms | 17.1 ms | 57.7 ms | 4,438 | compute |

Up to batch 64 the step doesn't slow down at all, so throughput grows 64×. The crossover is where compute time equals memory time. The parameter count cancels out:

```
B × 2 / 71e12 = 2 / 936e9
B = 71e12 / 936e9 ≈ 76
```

The general name for this is **operational intensity**: FLOPs per byte moved. At batch 1, decode does about 1 FLOP per byte (2 FLOPs per 2-byte weight). The 3090 needs about 76 FLOPs per byte to keep its arithmetic busy. That number is the **ridge point**.

## The roofline picture {#roofline}
> Plot speed against FLOPs per byte: a slanted line where memory limits you, a flat line where compute does. Every kernel lives under that roof.

```
FLOP/s
  |            ______ peak 71T
  |           /
  |          /  ← ridge ≈ 76
  |         /
  |        /  bandwidth × intensity
  |       /
  +-------------------- FLOP/byte
     decode, batch 1: far left
```

Left of the ridge, speed rises with intensity: you're memory-bound, so batching or reading fewer bytes helps. Right of it, you're at the compute ceiling, and only less arithmetic or faster hardware helps. The paper behind this picture is in this lesson's notes.

## When it's neither: kernel launches {#launches}
> Small models are often limited by the CPU sending work to the GPU. Each operation is a separate kernel launch, and launches have a fixed cost.

Your 30M model reads only about 120 MB of fp32 weights per step. At 936 GB/s that's about 0.13 ms, far below the roughly 2 ms a decode step actually takes. So where's the time?

Every PyTorch operation (a matmul, an add, a softmax) becomes a **kernel** that the CPU launches on the GPU, one after another, from Python. A decode step of the 30M model is about 70 launches, and each has a fixed overhead. The illustrative profile shows:

| Per decode step | Time |
|---|---|
| CPU launching kernels | 1.32 ms |
| GPU actually busy | 0.45 ms |
| Whole step | 2.08 ms |

The GPU is busy only 22% of the time. It's like making 70 tiny HTTP requests where one combined request would do: round-trip overhead dominates the actual work. The roofline doesn't model this at all, which is exactly why you profile.

## Timing a GPU honestly {#timing}
> GPU calls return before the work is done. Wait for the GPU before you read the clock, and throw away the first runs.

CUDA work is **asynchronous**. A call like `model(x)` queues kernels and returns immediately, a bit like starting an async request in Java and not waiting on its `Future`. If you read the clock straight after, you've timed the queueing, not the work.

So:

1. Call `torch.cuda.synchronize()` before starting the clock and again before stopping it.
2. Discard the first few iterations, which pay one-off costs such as memory allocation.
3. Change one thing at a time and re-time it.
4. After every speed-up, check the logits still match.

**torch.profiler** shows where time goes, operation by operation: CPU time, GPU time and the number of calls. Read the top rows before you optimize anything.

## torch.compile {#compile}
> torch.compile traces your model and generates fused kernels. With CUDA graphs, a whole step can be replayed with a single launch. The first call is slow because it compiles.

`torch.compile(model)` watches your model run, then generates new GPU code. It **fuses** chains of small operations, such as an add followed by a GELU, into one kernel, so there are fewer launches and fewer round trips through memory. With `mode="reduce-overhead"` it also uses **CUDA graphs**: it records the whole sequence of kernels once and replays it with a single launch, which is exactly the fix for a launch-bound model.

Things to know:

- **The first call is slow.** In the recording, compiling takes 38 seconds. Time steady-state speed separately.
- **Shapes should stay fixed.** CUDA graphs replay one recorded set of shapes. A cache that grows with `torch.cat` changes shape every step, so preallocate the cache at full length (2.2's hints).
- **Check the outputs.** Fused kernels can add numbers in a different order, so tiny differences are normal. The recording shows a maximum logit difference around 4 × 10⁻⁶ in fp32.

## SDPA and FlashAttention {#sdpa}
> SDPA computes ordinary attention. On a GPU it can pick a fused kernel that works in small tiles and never writes the full T × T score table to memory.

Plain attention builds a scores tensor of shape `[B, H, T, T]`. For one Llama-sized layer at 8k context:

```
1 × 32 × 8192 × 8192 × 2 bytes
= 4 GiB, for one layer
```

That's written to GPU memory and read back for the softmax and the mix. **FlashAttention** computes the same numbers in tiles that fit in the GPU's small, fast on-chip memory, keeping a running softmax per row, and never stores the full table. Same result, much less memory traffic. Your challenge checks the "same result" part to about 10⁻⁷.

Notes for the challenge, which uses 4-D tensors `[B, H, T, D]`:

- To score queries against keys you swap only the **last two** dimensions of k. On a tensor with more than 2 dims, `.T` reverses *all* of them, `[D, T, H, B]`, and the matmul fails.
- Scale by **√D**, not D and not 1.
- The mask keeps the **lower** triangle, keys 0…t for query t. Keeping the upper one hides the past and shows the future.
- Masked scores need `-inf`. A 0 is an ordinary score.
- Softmax runs over the **keys**, the last dimension. Over the queries, each key's weights would sum to 1 instead of each query's.

Why SDPA barely helps decode: with a KV cache there's one query per sequence, so the score table is one row, and there's little traffic to save. The win shows up in prefill. In the recording, prefill attention for 16 × 512 tokens drops from 4.1 to 1.6 ms. With one cached query, pass `is_causal=False`, since nothing after the newest token exists to hide.

## On your GPU {#gpu}
> You'll profile a few decode steps of your 30M model, then try SDPA, torch.compile and both. The recording is illustrative sample data, not a real run.

The replay shows `profile_generate.py`. Healthy output looks like:

| Mode | Tokens/s |
|---|---|
| eager | 480 |
| + SDPA | 495 |
| torch.compile | 1,040 |
| compile + SDPA | 1,090 |

- The profile blames kernel launches: GPU busy about 22% of each step.
- SDPA alone barely moves decode, as explained above.
- Compile with CUDA graphs roughly doubles it, for 2.27× overall.
- The same 256 tokens come out, with logits within about 4 × 10⁻⁶.

The numbers are sample data, labelled as such in the app. You record two metrics: the largest logit difference between compiled and eager runs (at most 0.001), and the speedup, compiled + SDPA tok/s ÷ eager tok/s (at least 1.3).

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **FLOP**: one floating-point operation. A multiply-add counts as 2.
- **TFLOPS**: 10¹² FLOPs per second, a GPU's peak arithmetic rate.
- **Memory bandwidth**: how many bytes per second the GPU can read from its main memory.
- **Compute-bound / memory-bound**: waiting on arithmetic, or on reading data.
- **Roofline**: speed limit = the lower of peak compute and bandwidth × operational intensity.
- **Operational intensity**: FLOPs per byte moved.
- **Ridge point**: the intensity where the two limits meet, about 76 FLOPs/byte on the 3090.
- **Kernel**: one function that runs on the GPU, such as a matmul.
- **Kernel launch**: the CPU telling the GPU to run a kernel. Each launch has a fixed overhead.
- **Launch-bound**: the GPU is idle because the CPU can't issue work fast enough.
- **Asynchronous**: a call returns before its work is done.
- **torch.cuda.synchronize()**: waits until all queued GPU work has finished.
- **torch.profiler**: records time per operation on the CPU and GPU.
- **torch.compile**: traces a model and generates fused kernels.
- **Kernel fusion**: merging several operations into one kernel.
- **CUDA graphs**: record a sequence of kernels once, replay it with one launch.
- **SDPA**: `F.scaled_dot_product_attention`, one call that can pick a fused attention kernel.
- **FlashAttention**: exact attention computed in tiles, without storing the T × T scores.
- **Eager mode**: normal PyTorch, running each operation as it's called.
- **Outputs unchanged after compile (max abs diff)**: this lesson's quality metric.
- **Speedup**: compiled + SDPA tokens/s ÷ eager tokens/s, this lesson's systems metric.
