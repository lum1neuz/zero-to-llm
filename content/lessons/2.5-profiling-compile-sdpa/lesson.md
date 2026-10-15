---
id: "2.5"
title: Profiling, compile, SDPA
summary: Find out what a decode step is waiting on before you speed it up. Do the roofline arithmetic, check fused attention against your own, then profile and compile your 30M model.
phase: "2"
hours: 5
type: run
status: ready
protected: false
cut: null
prerequisites:
  - "2.2"
papers:
  - title: FlashAttention
    authors: Dao et al.
    year: 2022
    url: https://arxiv.org/abs/2205.14135
  - title: "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
    authors: Williams, Waterman & Patterson
    year: 2009
    url: https://doi.org/10.1145/1498765.1498785
quality_metric:
  id: logits_match
  label: Outputs unchanged after compile
  unit: max abs diff
  inputs:
    - { id: max_diff, label: "Max |compiled − eager| logit difference", hint: "fp32, same prompt, printed by profile_generate.py" }
  compute: value
  reference: 0.0000038
  pass: { max: 0.001 }
systems_metric:
  id: speedup_x
  label: Speedup from torch.compile + SDPA
  unit: ×
  note: Decode tokens/s at batch 1 with the KV cache, compiled + SDPA divided by eager.
  inputs:
    - { id: eager_tok_s, label: eager tok/s, hint: From profile_generate.py }
    - { id: compiled_tok_s, label: torch.compile + SDPA tok/s, hint: Same run }
  compute: ratio
  reference: 2.27
  pass: { min: 1.3 }
quiz:
  - id: why_sync
    q: Why call torch.cuda.synchronize() before you read the clock?
    answer: 0
    options:
      - text: CUDA calls return before the GPU has finished, so without it you time the launch, not the work
        why: Kernels queue up and run asynchronously. synchronize() waits until the queue is empty, so the time you read covers the work you asked for.
      - text: It clears the GPU cache so every run starts cold
        why: It doesn't touch any cache. It only waits for queued work to finish.
      - text: It makes the GPU run faster
        why: It can only make your code wait. That's the point. You want to wait for the GPU before you stop the clock.
  - id: compile_warmup
    q: The first call to a torch.compile'd model takes 38 seconds and later calls take 1 ms. What happened in those 38 seconds?
    answer: 1
    options:
      - text: The weights were copied to the GPU again
        why: The weights were already on the GPU. A copy of 120 MB takes milliseconds, not seconds.
      - text: torch.compile traced the model and generated fused kernels, which later calls reuse
        why: Compilation happens once per input shape. Time it separately from steady-state speed, and keep shapes fixed (or mark them dynamic) so it doesn't recompile.
      - text: The GPU was warming up to its boost clock
        why: Clocks ramp up in milliseconds. 38 seconds is compilation.
  - id: flash_speed
    q: FlashAttention gives the same output as plain attention. Where does its speedup come from?
    answer: 2
    options:
      - text: It approximates softmax with a cheaper function
        why: It's exact. Your SDPA challenge matched plain attention to 1e-7.
      - text: It skips the attention heads that matter least
        why: Every head is computed. Nothing is skipped or pruned.
      - text: It works through tiles in fast on-chip memory and never writes the [T, T] scores to GPU memory
        why: Plain attention writes and re-reads a [B, H, T, T] tensor. Flash keeps a running softmax per tile and never stores the T×T matrix, so extra memory grows with T instead of T² and HBM traffic drops by a large constant factor.
---

## Goal

Before you make generation faster, find out what it's waiting on. Work out when a decode step is **memory-bound** and when it's **compute-bound** with roofline arithmetic. Check that `F.scaled_dot_product_attention` computes exactly the attention you wrote in 1.1. Then profile your 30M model on the GPU, apply SDPA and `torch.compile`, and measure the speedup while checking the logits haven't changed.

## Learn

```card
type: concept
id: where_time_goes
eyebrow: Where the time goes
title: At batch 1, the GPU mostly waits for memory.
visual:
  kind: bars
  max: 18
  bars:
    - { label: read weights, value: 17.1, text: 17.1 ms }
    - { label: do the math, value: 0.23, text: 0.23 ms }
body: |
  One decode step of Llama 3.1 8B in fp16 on an RTX 3090 reads all 16 GB of weights at 936 GB/s, which takes 17.1 ms. The math, 2 FLOPs per parameter at 71 TFLOPS, takes 0.23 ms. The step is **memory-bound**: a faster matmul wouldn't help, but reading fewer bytes (quantization, phase 3) or doing more work per read (batching, 2.4) would.
```

```card
type: calc
id: ridge_batch
eyebrow: Do the math
title: Up to what batch size is Llama 3.1 8B decode memory-bound on a 3090?
given:
  - { label: Peak compute, value: 71 TFLOPS (fp16) }
  - { label: Memory bandwidth, value: 936 GB/s }
  - { label: Per token, value: 2 FLOPs per parameter }
  - { label: Weights, value: "2 bytes per parameter, read once per step" }
unit: batch
answer: { min: 70, max: 80 }
placeholder: "0"
why_ok: "Compute time 2·P·B / 71×10¹² equals memory time 2·P / 936×10⁹ when B = 71×10¹² / 936×10⁹ ≈ 76. Below that, extra rows in the batch are nearly free."
why_wrong: "Set compute time equal to memory time. The parameter count cancels: B × 2 / 71×10¹² = 2 / 936×10⁹."
hint: Write compute time and memory time for a batch of B, set them equal, and solve for B. The parameter count cancels out.
```

```card
type: predict
id: scores_memory
eyebrow: Predict the output
title: How big is one layer's attention-score tensor at 8k context?
code: |
  B, H, T = 1, 32, 8192
  scores_bytes = B * H * T * T * 2   # fp16
  print(scores_bytes / 2**30, "GiB")
options:
  - text: 4.0 GiB
    correct: true
    why: 32 × 8192 × 8192 × 2 bytes = 4 GiB, for one layer of one sequence. Plain attention writes that to memory and reads it back. FlashAttention never stores it.
  - text: 0.125 GiB
    why: That's one head. There are 32 heads, each with its own T × T scores.
  - text: 1.0 GiB
    why: That's the whole KV cache for this sequence, all 32 layers (2.2). The scores are a T × T table per head, so they grow as T².
  - text: 128 GiB
    why: That's 32× too much. Multiply it out again, one factor at a time.
hint: Multiply it out. 8192 × 8192 is about 67 million.
```

```card
type: reorder
id: profile_loop
eyebrow: Put it in order
title: Arrange one round of performance work.
lines:
  - Time the baseline, with torch.cuda.synchronize() around the loop
  - Run torch.profiler over a few decode steps
  - Read the top rows to see where the time goes
  - Change one thing, like SDPA or torch.compile
  - Time it again, and check the logits still match
shuffle: [2, 4, 0, 3, 1]
why_ok: "Measure, look, change one thing, measure again. If you skip the profile, you optimize a guess. If you skip the logit check, you might be fast and wrong."
why_wrong: "You can't read a profile you haven't taken, or compare against a baseline you haven't timed."
hint: You need a number to beat before anything else, and the last step checks the change against it.
```

```card
type: quiz
ref: why_sync
```

```card
type: quiz
ref: compile_warmup
```

```card
type: quiz
ref: flash_speed
```

## Build steps

### Memory-bound or compute-bound?

```challenge
id: roofline
file: profile.py
minutes: 25
start: { code: "import torch\nimport torch.nn.functional as F\n\n" }
setup:
  - say: Two numbers describe a GPU for this lesson. How many floating-point operations it can do per second, and how many bytes per second it can read from memory. Here are the RTX 3090's.
    type: "rtx3090 = dict(peak_tflops=71, bandwidth_gbs=936)  # dense fp16 tensor-core peak (fp32 accumulate), memory bandwidth\n\n"
    pause: 3.0
  - say: "A decode step has to do its math and read every weight. It finishes when the slower of the two finishes."
    pause: 3.0
  - say: "Here's the stub. Your turn: count the FLOPs and bytes, then decide what the step is waiting on."
    type_file: roofline/stub.py
    pause: 1.2
tests: roofline/tests.py
solution: roofline/solution.py
alternatives:
  - roofline/alternatives/if_else.py
  - roofline/alternatives/ternary_flip.py
hints:
  nudge: Which of the two costs grows with the batch, and which one doesn't? And if both happen at once, when is the step over?
  concept: "Each token in the batch does a multiply and an add with every weight: 2 × params × batch FLOPs. The weights are read once per step and shared by every token: params × bytes_per_param bytes. Compute and memory overlap, so the step takes the larger of the two times, and the larger one names the bound."
  skeleton: |
    flops = _ * params * _____
    bytes_moved = ______ * _______________
    ms = ___(compute_ms, memory_ms)
    bound = "compute" if compute_ms _ memory_ms else "memory"
  solution: |
    flops = 2 * params * batch
    bytes_moved = params * bytes_per_param
    ms = max(compute_ms, memory_ms)
    bound = "compute" if compute_ms > memory_ms else "memory"
on_solve: "From batch 1 to 64 the step stays at 17.1 ms, so throughput grows 64×. At 256 it turns compute-bound and the step slows down. Your 30M model is so small that it's bound by neither: kernel launches dominate, as the profile in step 3 shows."
mistakes:
  - { file: roofline/mistakes/placeholders.py, fails: returns_pair, says: "`flops` is still `...`" }
  - { file: roofline/mistakes/bound_placeholder.py, fails: returns_pair, says: "`bound` is still `...`" }
  - { file: roofline/mistakes/summed.py, fails: step_time, says: "You added compute time and memory time" }
  - { file: roofline/mistakes/min.py, fails: step_time, says: "the *faster* of the two" }
  - { file: roofline/mistakes/bytes_per_token.py, fails: step_time, says: "reads the weights once per token" }
  - { file: roofline/mistakes/no_batch_flops.py, fails: step_time, says: "Your FLOPs don't grow with the batch" }
  - { file: roofline/mistakes/one_flop.py, fails: step_time, says: "You count 1 FLOP per parameter" }
  - { file: roofline/mistakes/bits.py, fails: step_time, says: "You multiplied by 8" }
  - { file: roofline/mistakes/bound_flipped.py, fails: names_bound, says: "The bound is the side that takes longer" }
  - { file: roofline/mistakes/no_main.py, fails: script_runs, says: "is missing" }
```

### Check SDPA against your attention

```challenge
id: sdpa_match
file: profile.py
minutes: 25
start: { from: roofline }
setup:
  - say: "F.scaled_dot_product_attention is one call that, on a GPU, picks a fused kernel like FlashAttention. Before you swap it into your model, check that it computes the same thing as the attention you wrote in 1.1."
    type_file: sdpa_match/stub.py
    pause: 2.6
  - say: "Your turn: write causal attention for four-dimensional q, k and v. The script prints the largest difference from SDPA."
    pause: 1.2
tests: sdpa_match/tests.py
solution: sdpa_match/solution.py
alternatives:
  - sdpa_match/alternatives/triu_mask.py
  - sdpa_match/alternatives/where_softmax.py
hints:
  nudge: "Three lines, same as 1.1: scores, mask, softmax-and-mix. With four dims, which two dims of k do you swap?"
  concept: "`k.transpose(-2, -1)` swaps only the last two dims, so `q @ k.transpose(-2, -1)` is [B, H, T, T]. Divide by √D. `torch.tril` of a T×T block of True marks the keys each query may see, so fill the rest with `-inf`. Softmax over the last dim (the keys), then `@ v`."
  skeleton: |
    scores = q @ k.transpose(__, __) / D**___
    masked = scores.masked_fill(~torch.____(torch.ones(T, T, dtype=torch.bool)), float("____"))
    return F.softmax(masked, dim=__) @ _
  solution: |
    scores = q @ k.transpose(-2, -1) / D**0.5
    masked = scores.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf"))
    return F.softmax(masked, dim=-1) @ v
on_solve: "Same output to about 1e-7. Your version builds the whole scores tensor, 8 KB here and 4 GiB per layer at 8k context on Llama. On a GPU, SDPA's flash kernel computes the same numbers tile by tile and never writes that tensor out. Same math, far less memory traffic."
mistakes:
  - { file: sdpa_match/mistakes/placeholders.py, fails: output_shape, says: "still returns `...`" }
  - { file: sdpa_match/mistakes/return_placeholder.py, fails: output_shape, says: "still returns `...`" }
  - { file: sdpa_match/mistakes/k_T.py, fails: output_shape, says: "reverses *all four* dims" }
  - { file: sdpa_match/mistakes/no_values.py, fails: output_shape, says: "the attention weights" }
  - { file: sdpa_match/mistakes/no_scale.py, fails: matches_sdpa, says: "the scores aren't scaled" }
  - { file: sdpa_match/mistakes/scale_by_d.py, fails: matches_sdpa, says: "divided by D = 4" }
  - { file: sdpa_match/mistakes/no_mask.py, fails: matches_sdpa, says: "Hide the future" }
  - { file: sdpa_match/mistakes/past_masked.py, fails: matches_sdpa, says: "hides the past and keeps the future" }
  - { file: sdpa_match/mistakes/zero_fill.py, fails: matches_sdpa, says: "0 is an ordinary score" }
  - { file: sdpa_match/mistakes/softmax_queries.py, fails: matches_sdpa, says: "runs over the queries" }
  - { file: sdpa_match/mistakes/no_main.py, fails: script_runs, says: "are missing" }
```

### Profile and compile your 30M model

```replay
id: profile_30m
intro: On your GPU, profile a few decode steps of your 30M model, then try SDPA, torch.compile and both. Watch a recording first, so you know what a launch-bound profile looks like.
log: replays/profile-30m.jsonl
speed: 40
recorded:
  hardware: RTX 3090
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: decode tokens per second, by mode
  flag_above: 100000
  x_labels: [eager, compile + SDPA]
  caption: SDPA barely moves decode, where each step has one query. torch.compile with CUDA graphs removes the launch overhead that the profile blamed.
summary:
  - { label: Bottleneck, value: kernel launches (GPU busy 22%) }
  - { label: Speed, value: "480 → 1,090 tok/s" }
  - { label: Speedup, value: 2.27× }
  - { label: Logit change, value: max diff 3.8e-6 }
run_yourself: |
  # on your machine or Colab, from the course repo
  python profile_generate.py --ckpt runs/tinystories-30m/best.pt \
    --new-tokens 256 --modes eager,sdpa,compile,compile+sdpa
```

### Record your metrics

```metrics
id: record
intro: Both numbers come from your own GPU run. Report steady-state speed, after torch.compile has warmed up.
```

## Hints

- Time with `torch.cuda.synchronize()` before both clock reads, and throw away the first few iterations. With `torch.compile`, throw away the whole compile call.
- `torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])` with `prof.key_averages().table(sort_by="cuda_time_total")` gives the table in the recording. Export a trace to see the gaps between kernels.
- `mode="reduce-overhead"` uses CUDA graphs, which need fixed shapes. Preallocate the KV cache at its full length (2.2's hints) so each step has the same shapes.
- In your model, swap the three attention lines for `F.scaled_dot_product_attention(q, k, v, is_causal=True)`. With a KV cache and one query, pass `is_causal=False`: the query is the newest token, so there's nothing to hide.

## Done when

- [ ] decode_step_ms finds the memory-bound and compute-bound regimes {#roofline auto=challenge:roofline}
- [ ] Your attention matches F.scaled_dot_product_attention {#sdpa_match auto=challenge:sdpa_match}
- [ ] Logit match and speedup recorded from your profiled GPU run {#metrics_recorded auto=metrics}
