---
id: "5.1"
title: vLLM vs llama.cpp
summary: Serve your Mongolian model with llama.cpp and vLLM, work out why decode is memory-bound, and measure who wins with 1 user and with 16.
phase: "5"
hours: 6
type: run
status: ready
protected: false
cut: 4
prerequisites:
  - "3.6"
papers: []
quality_metric:
  id: output_match
  label: Greedy outputs match across engines
  unit: "%"
  note: Share of 50 prompts whose first 64 greedy tokens are identical in both engines, both in fp16. Different kernels round differently and can flip a near-tie, so 100% is rare.
  inputs:
    - { id: match_pct, label: Prompts with identical greedy output (%), hint: "50 prompts, first 64 tokens, both engines fp16" }
  compute: value
  reference: 92
  pass: { min: 80 }
systems_metric:
  id: throughput_x
  label: vLLM throughput vs llama.cpp at 16 users
  unit: ×
  note: Output tokens per second summed over all 16 users, same prompts and max tokens in both engines.
  inputs:
    - { id: llama_tps, label: llama.cpp tok/s at 16 users, hint: From bench_serving.py }
    - { id: vllm_tps, label: vLLM tok/s at 16 users, hint: "Same benchmark, vLLM server" }
  compute: ratio
  reference: 1.92
  pass: { min: 1.3 }
quiz:
  - id: which_engine
    q: One user, a laptop with a 12 GB GPU, and your 8B model. Which setup fits?
    answer: 0
    options:
      - text: llama.cpp with a Q4_K_M GGUF
        why: About 4.9 GB of weights, so it fits with room for context, and llama.cpp can offload layers to the CPU if it ever doesn't. One user is exactly the case it's built for.
      - text: vLLM in fp16
        why: The fp16 weights alone are 15 GiB, more than the 12 GB card holds, and vLLM keeps everything on the GPU. It's built for a big GPU serving many users.
      - text: Either one. They run the same model, so they need the same memory
        why: The memory depends on the format. fp16 needs 2 bytes per weight; a 4-bit GGUF needs about 0.6. vLLM can load quantized checkpoints too, but it still wants the whole model plus KV blocks on the GPU.
  - id: batch_is_cheap
    q: From 1 user to 16, vLLM's total throughput rises from 48 to 612 tok/s, but each user's TPOT only goes from 20.6 to 26.3 ms. Why is the batch so cheap?
    answer: 1
    options:
      - text: vLLM reuses answers between users who ask similar things
        why: Every user gets their own tokens from their own forward passes. Nothing is copied between answers.
      - text: Each step reads the weights once, whether it makes 1 token or 16
        why: Decode is memory-bound. Streaming 15 GiB of weights dominates the step, and all 16 sequences use that same read. Extra users mostly add math, which the GPU had spare.
      - text: The GPU splits into 16 parts, one per user
        why: If each user got 1/16 of the GPU, each TPOT would be about 16× longer. It barely moved, because the users share the expensive part of the step.
  - id: greedy_mismatch
    q: Both engines run fp16 with greedy decoding, yet 4 of 50 outputs differ. What's the most likely cause?
    answer: 2
    options:
      - text: One engine has a bug
        why: A bug usually shows up as garbage or a mismatch from the first token. Four late splits at near-ties are what normal numeric differences look like.
      - text: Greedy decoding is random, so outputs never repeat
        why: Greedy always takes the top token, so one engine repeats itself exactly. The differences are between engines.
      - text: The kernels add numbers in a different order, and a near-tie flips
        why: Floating-point addition depends on order. When two tokens' logits are within a rounding error, the engines can pick different ones, and every token after that follows a different path.
---

## Goal

Serve your Mongolian 8B model with **llama.cpp** and with **vLLM**, and benchmark both with 1, 4 and 16 users. First, in the browser, work out the arithmetic that decides the result: why one user's decode is limited by memory bandwidth, why a batch of users rides almost free, and how to read a benchmark's TTFT, TPOT and throughput correctly.

## Learn

```card
type: concept
id: two_engines
eyebrow: Two engines
title: Tied at one user, apart at sixteen.
visual:
  kind: bars
  max: 650
  bars:
    - { label: llama.cpp ×1, value: 50, text: "50" }
    - { label: vLLM ×1, value: 48, text: "48" }
    - { label: llama.cpp ×16, value: 318, text: "318" }
    - { label: vLLM ×16, value: 612, text: "612" }
body: |
  **llama.cpp** is one C++ binary that runs GGUF files on a CPU, a Mac or a GPU, with k-quants down to 2 bits. **vLLM** is a Python server for big GPUs that keeps many users' requests in one batch. Both run the same model. The bars are total tok/s for an 8B fp16 model on an RTX 3090 (illustrative numbers): with one user they tie, with sixteen vLLM makes about twice as many tokens.
```

```card
type: calc
id: batch1_ceiling
eyebrow: Do the math
title: Every decode step reads all the weights once. What's the fastest one user can get tokens?
given:
  - { label: Parameters, value: 8 billion }
  - { label: Bytes per parameter, value: 2 (fp16) }
  - { label: GPU memory bandwidth, value: 936 GB/s }
unit: tok/s
answer: { min: 57, max: 60 }
placeholder: "0"
why_ok: "16 GB per step at 936 GB/s is 17.1 ms, so at most 58.5 tokens per second. Real engines get 48–50: close to the ceiling, so one user's speed is set by memory bandwidth, not by the GPU's math."
why_wrong: "One token needs one full read of the weights: 8e9 × 2 bytes. Divide the bandwidth by the bytes per step to get steps, and tokens, per second."
hint: How many bytes does one step read? How many of those reads fit into one second at 936 GB/s?
```

```card
type: fill_blank
id: roofline_blanks
eyebrow: Fill the blanks
title: Complete the step-time model.
code: |
  mem_ms = n_params * bytes_per_param / BANDWIDTH * 1000
  compute_ms = 2 * n_params * [[0]] / PEAK_FLOPS * 1000
  step = [[1]](mem_ms, compute_ms)
  tok_per_s = batch * 1000 / [[2]]
tokens: [batch, bytes_per_param, max, min, step, mem_ms]
answer: [batch, max, step]
tips:
  bytes_per_param: The math doesn't care how a weight is stored. FLOPs grow with how many sequences each weight multiplies.
  min: The faster job finishes and then waits for the slower one, so the slower one sets the step time.
  mem_ms: Once the math is the bottleneck, the step takes longer than the weight read. Use the real step time.
why_ok: "Memory time is fixed per step, math time grows with the batch, and the step lasts as long as the slower one. Each step makes `batch` tokens."
why_wrong: "Ask what grows when more users join: the bytes you read, or the math you do? Then: which job does the GPU wait for?"
hint: Every sequence in the batch uses the same weights, but each needs its own multiply-adds.
```

```card
type: predict
id: ttft_tpot
eyebrow: Predict the output
title: One request from a benchmark log. What prints?
code: |
  r = {"arrive": 1.0, "first": 1.3, "done": 3.3, "tokens": 101}
  ttft = r["first"] - r["arrive"]
  tpot_ms = (r["done"] - r["first"]) / (r["tokens"] - 1) * 1000
  print(f"{ttft:.1f} s, {tpot_ms:.1f} ms")
options:
  - text: "0.3 s, 20.0 ms"
    correct: true
    why: "It waited 0.3 s for its first token. The other 100 tokens arrived over the next 2.0 s: 20 ms each."
  - text: "1.3 s, 20.0 ms"
    why: That's the first token's timestamp, counted from the start of the run. TTFT counts from when this request arrived.
  - text: "0.3 s, 19.8 ms"
    why: That spreads 2.0 s over all 101 tokens. The first token arrived at the start of that window, so only 100 tokens share it.
  - text: "0.3 s, 23.0 ms"
    why: That's (3.3 − 1.0) / 100, which counts the 0.3 s wait as streaming time. TTFT already measures the wait.
hint: TTFT runs from arrive to first. TPOT spreads the time from first to done over the tokens that came after the first.
```

```card
type: quiz
ref: which_engine
```

```card
type: quiz
ref: batch_is_cheap
```

```card
type: quiz
ref: greedy_mismatch
```

## Build steps

### How fast can one GPU decode?

```challenge
id: roofline
file: serving_math.py
minutes: 30
start: { code: "" }
setup:
  - say: "Every decode step reads all of the model's weights from GPU memory and does the math for every sequence in the batch. Start with the GPU's two limits."
    type: "# RTX 3090, rounded spec-sheet numbers\nBANDWIDTH = 936e9    # bytes per second from GPU memory\nPEAK_FLOPS = 71e12   # fp16 tensor-core FLOPs per second\n\n"
    pause: 2.6
  - say: "The GPU streams weights in and multiplies at the same time. A step is over when both jobs are, so whichever takes longer sets the pace."
    pause: 3.4
  - say: "Here's the stub. Fill in step_ms and tokens_per_s, then run the table for an 8B model in fp16."
    type_file: roofline/stub.py
    pause: 1.2
tests: roofline/tests.py
solution: roofline/solution.py
alternatives:
  - roofline/alternatives/if_else.py
  - roofline/alternatives/seconds_first.py
  - roofline/alternatives/flops_per_token.py
hints:
  nudge: Two clocks run at once, one for moving bytes and one for doing math. Work out each in ms, then ask which one the GPU waits for.
  concept: "Reading the weights moves `n_params × bytes_per_param` bytes at `BANDWIDTH` bytes per second, and the batch doesn't change it: every sequence uses the same weights. The math is a multiply and an add per parameter per token, so `2 × n_params` FLOPs per sequence, times the batch, at `PEAK_FLOPS`. Seconds × 1000 gives ms. The step takes the larger of the two, and each step makes `batch` tokens."
  skeleton: |
    mem_ms = n_params * _______________ / BANDWIDTH * ____
    compute_ms = ___ * n_params * _____ / PEAK_FLOPS * 1000
    return ___(mem_ms, compute_ms)
    return _____ * 1000 / step_ms(n_params, bytes_per_param, batch)
  solution: |
    mem_ms = n_params * bytes_per_param / BANDWIDTH * 1000
    compute_ms = 2 * n_params * batch / PEAK_FLOPS * 1000
    return max(mem_ms, compute_ms)
    return batch * 1000 / step_ms(n_params, bytes_per_param, batch)
on_solve: "Read the table. From batch 1 to 64 the step stays at 17.09 ms, so throughput climbs exactly 64× (58 → 3,744 tok/s). Past batch 76 the math takes over and throughput flattens. Real engines hit the knee later (attention also reads every sequence's KV cache, and kernels don't reach peak FLOPs), but the shape holds: serving many users at once is how a GPU earns its keep."
mistakes:
  - { file: roofline/mistakes/placeholders.py, fails: mem_bound_at_1, says: "`step_ms` still returns `...`" }
  - { file: roofline/mistakes/seconds.py, fails: mem_bound_at_1, says: "That's seconds" }
  - { file: roofline/mistakes/no_bytes.py, fails: mem_bound_at_1, says: "each fp16 parameter is 2 bytes" }
  - { file: roofline/mistakes/min.py, fails: mem_bound_at_1, says: "Which of the two sets the pace?" }
  - { file: roofline/mistakes/sum.py, fails: mem_bound_at_1, says: "at the same time" }
  - { file: roofline/mistakes/mem_times_batch.py, fails: batch_shares_weights, says: "share one read of the weights" }
  - { file: roofline/mistakes/compute_no_batch.py, fails: compute_bound_at_256, says: "as if the math were free" }
  - { file: roofline/mistakes/return_mem_only.py, fails: compute_bound_at_256, says: "does your return look at it?" }
  - { file: roofline/mistakes/one_flop.py, fails: compute_bound_at_256, says: "a multiply *and* an add" }
  - { file: roofline/mistakes/tps_no_batch.py, fails: tokens_per_s, says: "the same rate as one user" }
  - { file: roofline/mistakes/tps_per_ms.py, fails: tokens_per_s, says: "tokens per *millisecond*" }
  - { file: roofline/mistakes/no_loop.py, fails: script_runs, says: "is missing" }
```

### Read a benchmark like a serving engineer

```challenge
id: bench_stats
file: bench.py
minutes: 30
start: { code: "import statistics\n\n" }
setup:
  - say: "A serving benchmark records four things per request: when it arrived, when its first token came back, when it finished, and how many tokens it got."
    type: "# One benchmark run: 7 requests against your server, times in seconds from the start of the run\nrequests = [\n    {\"arrive\": 0.00, \"first\": 0.18, \"done\": 2.58, \"tokens\": 121},\n    {\"arrive\": 0.05, \"first\": 0.24, \"done\": 1.92, \"tokens\": 81},\n    {\"arrive\": 0.40, \"first\": 0.61, \"done\": 4.13, \"tokens\": 161},\n    {\"arrive\": 1.10, \"first\": 1.29, \"done\": 2.97, \"tokens\": 81},\n    {\"arrive\": 1.90, \"first\": 2.95, \"done\": 4.27, \"tokens\": 61},\n    {\"arrive\": 2.00, \"first\": 2.21, \"done\": 4.85, \"tokens\": 121},\n    {\"arrive\": 2.70, \"first\": 2.90, \"done\": 4.50, \"tokens\": 81},\n]\n\n"
    pause: 2.8
  - say: "Users feel two numbers: how long until text starts, time to first token, and how fast it streams after that, time per output token. The operator watches a third: tokens per second across everyone."
    pause: 4.0
  - say: "Here's the stub. Fill in the four values, then run it."
    type_file: bench_stats/stub.py
    pause: 1.2
tests: bench_stats/tests.py
solution: bench_stats/solution.py
alternatives:
  - bench_stats/alternatives/sorted_middle.py
  - bench_stats/alternatives/loops.py
  - bench_stats/alternatives/named_parts.py
hints:
  nudge: "Each number has its own start and stop. TTFT runs from arriving to the first token. TPOT runs from the first token to the last. Throughput covers the whole run."
  concept: "`r[\"first\"] - r[\"arrive\"]` is one request's TTFT. After the first token, `tokens − 1` more arrive between `first` and `done`, so TPOT is `(done − first) / (tokens − 1)`, × 1000 for ms. Requests overlap, so throughput divides all the tokens by the run's wall-clock span (last `done` minus first `arrive`), not by the sum of their durations. p50 is the median: `statistics.median`."
  skeleton: |
    ttft = [r["_____"] - r["______"] for r in requests]
    tpot = [(r["____"] - r["_____"]) / (r["tokens"] - _) * ____ for r in requests]
    tok_per_s = sum(r["tokens"] for r in requests) / (___(r["done"] for r in requests) - ___(r["arrive"] for r in requests))
    ttft_p50 = statistics.______(ttft)
  solution: |
    ttft = [r["first"] - r["arrive"] for r in requests]
    tpot = [(r["done"] - r["first"]) / (r["tokens"] - 1) * 1000 for r in requests]
    tok_per_s = sum(r["tokens"] for r in requests) / (max(r["done"] for r in requests) - min(r["arrive"] for r in requests))
    ttft_p50 = statistics.median(ttft)
on_solve: "TTFT p50 0.20 s, TPOT 21.1 ms, 146 tok/s. Now look at request 4: it waited 1.05 s for its first token, five times the others, because it arrived while the server was busy. The median hides it. That's why serving dashboards also show p90 and p99: try `statistics.quantiles(ttft, n=10)[-1]` inside `summarize` for p90."
mistakes:
  - { file: bench_stats/mistakes/placeholders.py, fails: ttft_p50, says: "`tpot` is still `...`" }
  - { file: bench_stats/mistakes/ttft_from_start.py, fails: ttft_p50, says: "you're timing it from 0 s" }
  - { file: bench_stats/mistakes/ttft_mean.py, fails: ttft_p50, says: "That's the mean" }
  - { file: bench_stats/mistakes/ttft_unsorted.py, fails: ttft_p50, says: "in arrival order" }
  - { file: bench_stats/mistakes/ttft_whole_request.py, fails: ttft_p50, says: "TTFT stops at the *first* token" }
  - { file: bench_stats/mistakes/ttft_list.py, fails: ttft_p50, says: "is a list" }
  - { file: bench_stats/mistakes/tpot_div_tokens.py, fails: tpot, says: "Only the tokens after the first" }
  - { file: bench_stats/mistakes/tpot_seconds.py, fails: tpot, says: "which is seconds" }
  - { file: bench_stats/mistakes/tpot_includes_wait.py, fails: tpot, says: "counts the wait for the first token" }
  - { file: bench_stats/mistakes/tps_from_zero.py, fails: throughput, says: "Start the clock at the first arrival" }
  - { file: bench_stats/mistakes/tps_sum_durations.py, fails: throughput, says: "they overlap" }
  - { file: bench_stats/mistakes/tps_mean_rates.py, fails: throughput, says: "how fast one user sees tokens" }
  - { file: bench_stats/mistakes/no_main.py, fails: script_runs, says: "are missing" }
```

### Benchmark both engines

```replay
id: bench_engines
intro: Converting the model, starting both servers and running the benchmark takes about an hour on the GPU. Watch a sample run first so you know what the output looks like.
log: replays/bench-1-4-16.jsonl
speed: 40
recorded:
  hardware: RTX 3090 24 GB
  illustrative: true
  note: Sample data written for the course, not a real run. The shape is typical, but your numbers will differ. Replace it with a recording of your own run.
chart:
  title: TTFT p50 per run (ms)
  flag_above: 300
  x_labels: [llama.cpp ×1, vLLM ×16]
  caption: "With one user the engines are nearly tied: both are memory-bound on the same 15 GiB of weights. The gap opens with load. At 16 users vLLM keeps first tokens fast and makes about twice the tokens. It was built for this case, and paged KV memory (5.2) and a scheduler that admits requests every step (5.4) are two of the pieces behind it."
summary:
  - { label: TTFT p50 at 16 users, value: 436 → 97 ms }
  - { label: Throughput at 16 users, value: 318 → 612 tok/s (1.92×) }
  - { label: Greedy match, value: 46 / 50 prompts (92%) }
run_yourself: |
  # on your GPU machine, from the course repo
  python llama.cpp/convert_hf_to_gguf.py runs/mn-8b --outtype f16 --outfile runs/mn-8b/mn-8b-f16.gguf
  llama-server -m runs/mn-8b/mn-8b-f16.gguf -ngl 99 -c 16384 -np 16 --port 8080
  # stop llama-server, then
  vllm serve runs/mn-8b --dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.92 --port 8000
  python bench_serving.py --prompts mn_prompts.jsonl --users 1,4,16 --max-tokens 256 \
    --engines llama.cpp:8080,vllm:8000 --greedy-check 64
```

### Record your metrics

```metrics
id: record
intro: One quality metric (do the engines agree?) and one systems metric (how much more vLLM serves at 16 users), from your own run.
```

## Hints

- Run the two servers one after the other. vLLM claims `--gpu-memory-utilization` of the whole GPU at startup for weights and KV blocks, so llama.cpp won't fit beside it.
- Use the same prompts, the same max tokens, greedy decoding and the same chat template in both engines. llama.cpp reads the template stored in the GGUF, so check it matches the one vLLM uses.
- llama.cpp splits `-c` across its `-np` slots: `-c 16384 -np 16` gives each user 1,024 tokens of context.
- If vLLM won't start with "not enough KV cache", lower `--max-model-len`. With 15 GiB of fp16 weights on a 24 GB card, about 4.4 GiB is left for KV blocks.
- Warm both servers up with a few requests before you measure. The first requests pay for CUDA graph capture and memory allocation.

## Done when

- [ ] Your roofline finds where decode turns compute-bound {#roofline auto=challenge:roofline}
- [ ] Your benchmark summary gets TTFT, TPOT and throughput right on a fresh run {#bench_stats auto=challenge:bench_stats}
- [ ] Both engines serve your model, benchmarked with 1, 4 and 16 users {#benchmarked}
- [ ] Greedy match and the 16-user throughput ratio recorded {#metrics_recorded auto=metrics}
