---
id: "0.1"
title: Environment check and baseline
summary: Prove your GPU does the work by timing a matmul, and compute the loss of a model that knows nothing, the number every later model has to beat.
phase: "0"
hours: 2
type: run
status: ready
protected: false
cut: null
prerequisites:
  - "0.0"
papers: []
quality_metric:
  id: baseline_loss
  label: Baseline char-model loss
  unit: nats
  note: The uniform model over 27 symbols, computed over all of names.txt. It should equal ln 27.
  inputs:
    - { id: baseline_loss, label: Uniform baseline on names.txt (nats), hint: The last line of env_check.py }
  compute: value
  reference: 3.2958
  pass: { min: 3.29, max: 3.30 }
systems_metric:
  id: matmul_tflops
  label: Measured matmul throughput
  unit: TFLOP/s
  note: Your best fp16 or bf16 number. The reference comes from the illustrative RTX 3090 replay. Any recent NVIDIA GPU clears 5; a CPU manages about 1.
  inputs:
    - { id: matmul_tflops, label: Best fp16/bf16 TFLOP/s, hint: From env_check.py }
  compute: value
  reference: 68.4
  pass: { min: 5 }
quiz:
  - id: loss_above_baseline
    q: Your first training step on the names data prints a loss of 27.9. What does that tell you?
    answer: 0
    options:
      - text: The model is confidently wrong, far worse than guessing uniformly
        why: "Knowing nothing costs ln 27 ≈ 3.30. A loss of 27.9 means the initial logits are huge, so the model is sure of the wrong answers. Lesson 0.4 fixes the initialization."
      - text: Nothing unusual. Loss always starts high and comes down
        why: A well-initialized model starts near the uniform baseline, 3.30 here. Anything far above it is wasted training steps spent undoing a bad init.
      - text: The data loader is broken
        why: A broken loader usually shows up as a loss that won't come down, or one that drops too fast. A starting loss far above ln 27 points at the initialization.
  - id: cuda_false
    q: "`torch.cuda.is_available()` prints False on a machine with an NVIDIA GPU, and `nvidia-smi` works. What's the most likely cause?"
    answer: 1
    options:
      - text: The GPU is too old for PyTorch
        why: Possible for a very old card, but rare. If the driver works, check the PyTorch build before blaming the hardware.
      - text: A CPU-only build of PyTorch is installed
        why: "`torch.version.cuda` prints None on a CPU-only build. Uninstall it and reinstall with the CUDA command from pytorch.org's selector."
      - text: The code forgot `.to('cuda')`
        why: "`is_available()` asks the PyTorch build and the driver, not your tensors. Moving tensors comes later."
---

## Goal

Two numbers before you train anything. First, how fast your GPU multiplies matrices, so you know it's really doing the work and how much faster fp16 and bf16 are than fp32. Second, the loss of a model that knows nothing: on 27 symbols that's ln 27 ≈ 3.30 nats. Every model in phase 0 is judged by how far below that it gets.

## Learn

```card
type: concept
id: two_numbers
eyebrow: Why check
title: Two numbers to check before you train.
visual:
  kind: bars
  max: 70
  bars:
    - { label: CPU fp32, value: 0.8, text: 0.8 TFLOP/s }
    - { label: GPU fp32, value: 22.1, text: "22.1" }
    - { label: GPU fp16, value: 68.4, text: "68.4" }
    - { label: GPU bf16, value: 67.7, text: "67.7" }
body: |
  The bars time the same big grid multiplication on your computer's main chip (CPU) and on its graphics card (GPU), in three number formats. Taller is faster. The numbers are illustrative, for an RTX 3090.

  If your GPU bar looks like the CPU bar, your graphics card isn't doing the work, and later lessons will crawl.

  The second number is the **baseline**: the score of a model that knows nothing and gives all 27 symbols the same chance. It's about 3.30. A model that has learned something scores lower.
```

```card
type: calc
id: uniform_loss
eyebrow: Do the math
title: What loss does a model that knows nothing get?
given:
  - { label: Symbols, value: "27 (a–z and \".\")" }
  - { label: "p(each)", value: "1 / 27" }
  - { label: Loss, value: "−ln p, averaged" }
unit: nats
answer: { min: 3.29, max: 3.30 }
placeholder: "0.00"
why_ok: "Every prediction gets p = 1/27, so every term is −ln(1/27) = ln 27 ≈ 3.2958, and so is their average."
why_wrong: "Every term in the average is the same: −ln(1/27). That's ln 27."
hint: "−ln(1/x) = ln x, and ln 27 is a little more than ln 20 ≈ 3.0."
```

```card
type: fill_blank
id: flops_formula
eyebrow: Fill the blanks
title: Turn a timing into TFLOP/s.
code: |
  # a and b are n×n, and secs is the time for one a @ b
  flops = [[0]] * n**3
  tflops = flops / secs / [[1]]
tokens: ["2", "1", "3", "1e12", "1e9"]
answer: ["2", "1e12"]
tips:
  "1": "Each of the n³ steps in a matmul is a multiply *and* an add, which is two floating-point operations."
  "3": "n³ already counts the steps. Each step is a multiply and an add."
  "1e9": "1e9 gives GFLOP/s. Tera is 10¹²."
why_ok: "An n×n matmul does n³ multiply-adds, so 2·n³ FLOPs. Divide by seconds for FLOP/s, then by 10¹² for TFLOP/s."
why_wrong: Count a multiply-add as two operations, and remember that tera means 10¹².
hint: "Each output entry is a dot product of length n: n multiplies and about n adds. There are n² entries."
```

```card
type: calc
id: tflops_from_ms
eyebrow: Do the math
title: One fp16 4096×4096 matmul takes 2.01 ms. How many TFLOP/s?
given:
  - { label: n, value: "4096" }
  - { label: FLOPs, value: "2 · n³ ≈ 1.374 × 10¹¹" }
  - { label: Time, value: 2.01 ms }
unit: TFLOP/s
answer: { min: 67, max: 70 }
placeholder: "0"
why_ok: "1.374 × 10¹¹ / 0.00201 s ≈ 6.84 × 10¹³ FLOP/s, which is 68.4 TFLOP/s."
why_wrong: "Convert ms to seconds first (2.01 ms = 0.00201 s), then divide the FLOPs by it and by 10¹²."
hint: "137.4 GFLOP in 2 ms is about 70 GFLOP per ms. How many ms are in a second?"
```

```card
type: predict
id: async_timing
eyebrow: Predict the meaning
title: This prints 0.00004. What happened?
code: |
  a = torch.randn(4096, 4096, device="cuda")
  t0 = time.perf_counter()
  b = a @ a
  print(time.perf_counter() - t0)
options:
  - text: The GPU did 137 GFLOP in 40 µs
    why: That would be about 3,400 TFLOP/s, roughly fifty times what an RTX 3090 manages and more than ten times any consumer GPU. The number is too good to be true.
  - text: "CUDA calls are asynchronous: the clock stopped when the matmul was queued, not when it finished"
    correct: true
    why: "Python hands the work to the GPU and moves on. Call `torch.cuda.synchronize()` before reading the clock, and time several runs after a warm-up."
  - text: PyTorch cached the result from an earlier run
    why: PyTorch doesn't cache results. The work is queued and still running when the clock is read.
hint: How long does Python wait after it hands a job to the GPU?
```

```card
type: quiz
ref: loss_above_baseline
```

```card
type: quiz
ref: cuda_false
```

## Build steps

### Compute the uniform baseline

```challenge
id: uniform_baseline
file: baseline.py
minutes: 15
start: { code: "import torch\n\n" }
setup:
  - say: "A few names, and a vocabulary of 27: the 26 letters plus \".\", which marks where a name ends."
    type: "words = [\"emma\", \"olivia\", \"ava\", \"isabella\", \"sophia\"]\nvocab_size = 27  # a to z, plus \".\" for the start and end of a name\n\n"
    pause: 2.6
  - say: "A model's loss is how surprised it is by the right answers. For each prediction, take the probability it gave the character that actually came next."
    pause: 3.4
  - say: "Your turn: write mean_nll, then run it on a model that gives every symbol the same chance."
    type_file: uniform_baseline/stub.py
    pause: 1.2
tests: uniform_baseline/tests.py
solution: uniform_baseline/solution.py
alternatives:
  - uniform_baseline/alternatives/torch_log.py
  - uniform_baseline/alternatives/sum_over_n.py
  - uniform_baseline/alternatives/neg.py
hints:
  nudge: Three moves. Take the log of each probability, average them, and fix the sign so that lower means better.
  concept: "The log of a probability is 0 when the model was certain and right, and very negative when it was badly wrong. Averaging makes the loss independent of how many predictions you made. The minus sign turns it into something you minimise."
  skeleton: |
    return -probs.____().____()
  solution: |
    return -probs.log().mean()
on_solve: "3.2958, whatever the names. Run the bigram model in 0.3 against this line: it gets to about 2.45. Now change one entry of probs to 0.9 and see the loss drop a little."
mistakes:
  - { file: uniform_baseline/mistakes/placeholder.py, fails: returns_number, says: "still returns `...`" }
  - { file: uniform_baseline/mistakes/no_mean.py, fails: returns_number, says: "one per prediction" }
  - { file: uniform_baseline/mistakes/no_minus.py, fails: uniform_is_ln27, says: "What flips it?" }
  - { file: uniform_baseline/mistakes/log2.py, fails: uniform_is_ln27, says: "That's in bits" }
  - { file: uniform_baseline/mistakes/summed.py, fails: uniform_is_ln27, says: "would the model be twice as bad?" }
  - { file: uniform_baseline/mistakes/no_log.py, fails: uniform_is_ln27, says: "from the *log* of each probability" }
  - { file: uniform_baseline/mistakes/log_of_mean.py, fails: average_of_logs, says: "the log of their average" }
```

### Check your GPU

```replay
id: env_check
intro: The environment check takes about a minute on your machine. Here's an illustrative run so you know what healthy output looks like.
log: replays/env-check.jsonl
speed: 30
recorded:
  hardware: RTX 3090
  illustrative: true
  note: Sample output written for the course, not a recording. Your numbers will differ; what matters is that the GPU is found and fp16/bf16 run several times faster than fp32.
chart:
  title: Time per 4096×4096 matmul (ms)
  flag_above: 10
  x_labels: [float32, bfloat16]
  caption: Lower is better. fp16 and bf16 use tensor cores and run about 3× faster than fp32. A red bar here means the matmul is barely faster than a CPU's 182 ms.
summary:
  - { label: GPU, value: RTX 3090 · 24 GB }
  - { label: Best matmul, value: 68.4 TFLOP/s (fp16) }
  - { label: Uniform baseline, value: 3.2958 nats }
run_yourself: |
  # env_check.py. Download names.txt first:
  # https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
  import time
  import torch

  print("torch", torch.__version__, "· CUDA", torch.version.cuda)
  assert torch.cuda.is_available(), "No GPU: install the CUDA build from pytorch.org"
  print(torch.cuda.get_device_name(0), "· bf16:", torch.cuda.is_bf16_supported())

  def bench(device, dtype, n=4096, reps=20):
      a = torch.randn(n, n, device=device, dtype=dtype)
      b = torch.randn(n, n, device=device, dtype=dtype)
      for _ in range(3):
          a @ b  # warm-up
      if device == "cuda":
          torch.cuda.synchronize()
      t0 = time.perf_counter()
      for _ in range(reps):
          a @ b
      if device == "cuda":
          torch.cuda.synchronize()  # GPU calls are async: wait before reading the clock
      secs = (time.perf_counter() - t0) / reps
      print(f"{device:5s} {str(dtype):15s} {secs * 1e3:8.2f} ms {2 * n**3 / secs / 1e12:6.1f} TFLOP/s")

  bench("cpu", torch.float32, reps=3)
  for dtype in (torch.float32, torch.float16, torch.bfloat16):
      bench("cuda", dtype)

  words = open("names.txt").read().splitlines()
  n = sum(len(w) + 1 for w in words)
  probs = torch.full((n,), 1 / 27)
  print(len(words), "names ·", n, "predictions · baseline", -probs.log().mean().item())
```

### Record your metrics

```metrics
id: record
intro: The uniform baseline from your own run over names.txt, and your best fp16 or bf16 matmul throughput.
```

## Hints

- `torch.version.cuda` prints `None` on a CPU-only build. Reinstall with the CUDA command from pytorch.org, inside your virtual environment.
- No GPU at home? Colab's free T4 is enough for phases 0–2. Turing cards like the T4 have no bf16 tensor cores, so use fp16 there.
- Laptop GPUs throttle when they heat up. If your numbers jump around, raise `reps` and run the script twice.
- The first CUDA call in a process is slow while it loads kernels. That's what the warm-up runs are for.

## Done when

- [ ] mean_nll gives ln 27 for the uniform model {#baseline auto=challenge:uniform_baseline}
- [ ] env_check.py runs on your machine: the GPU is found, and fp16 or bf16 runs several times faster than fp32 {#env_check}
- [ ] Baseline loss and matmul throughput recorded {#metrics_recorded auto=metrics}
