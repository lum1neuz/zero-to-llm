---
id: "1.4"
title: TinyStories 30M run
summary: Size a 30M-parameter GPT and its compute bill by hand, then train it on TinyStories, the model every quantization lesson in phase 3 works on.
phase: "1"
hours: 8
type: run
status: ready
protected: false
cut: 3
prerequisites:
  - "1.3"
papers:
  - title: TinyStories
    authors: Eldan & Li
    year: 2023
    url: https://arxiv.org/abs/2305.07759
quality_metric:
  id: val_ppl
  label: TinyStories validation perplexity
  unit: ppl
  note: exp(val loss) on the held-out stories, with the course's 21.5k-token BPE.
  inputs:
    - { id: val_ppl, label: Val perplexity, hint: "exp(val loss), printed when the run ends" }
  compute: value
  reference: 7.84
  pass: { max: 10.0 }
systems_metric:
  id: gpu_hours
  label: GPU hours to converge
  unit: h
  note: Wall-clock hours on one GPU to the final checkpoint. About 0.7 on an RTX 3090, 3–4 on a Colab T4.
  inputs:
    - { id: gpu_hours, label: GPU hours, hint: The run's wall clock in hours }
  compute: value
  reference: 0.71
  pass: { max: 5.0 }
quiz:
  - id: val_gap
    q: The run ends with train loss 1.98 and val loss 2.06. What does that tell you?
    answer: 1
    options:
      - text: The model is overfitting badly
        why: A gap of 0.08 is small. Overfitting shows up as val loss rising while train loss keeps falling, and here both are still going down together.
      - text: It generalises; half an epoch in, it has hardly seen any story twice
        why: The run covers 0.52 of an epoch, so almost every training batch is new text. Train and val loss stay close because the model can't memorise what it has seen once.
      - text: The validation set must contain different kinds of stories
        why: Val is a random slice of the same stories. A small gap is exactly what a healthy run on one distribution looks like.
      - text: Training should have stopped at step 5,000
        why: Val loss was still falling at every eval, from 2.29 at step 5,000 to 2.06 at the end. Stopping early would have left that on the table.
  - id: mfu
    q: The run gets 17.4 TFLOPS out of a GPU rated at 71. Where does the rest go?
    answer: 0
    options:
      - text: A 30M model's matmuls are small, so time goes to memory traffic, small kernels and the data loader instead of tensor-core math
        why: A 512-wide matmul finishes before the GPU is properly busy, and LayerNorm, softmax and GELU are limited by memory, not math. Bigger models reach 40–50% MFU on the same card.
      - text: The GPU throttles itself to a quarter of its speed to stay cool
        why: Thermal throttling costs a few percent at most. The gap here comes from how the work is shaped.
      - text: Only the backward pass runs on the tensor cores
        why: Forward and backward matmuls both run on tensor cores in bf16.
      - text: The 71 TFLOPS is for int8, so bf16 can only reach a quarter of it
        why: 71 TFLOPS is the RTX 3090's dense bf16 figure with fp32 accumulation. The shortfall comes from the model's size.
---

## Goal

Train the model every lesson in phase 3 quantizes: a **30M-parameter GPT** on **TinyStories**. First count its parameters by hand from the config, so you know what you're about to train. Then estimate the **compute** with the 6·N·D rule and turn it into GPU-hours. Then train it and record its validation perplexity and how long it took.

Behind schedule? This lesson is third in the cut order. If you skip it, 3.1 and 3.2 quantize the course's reference TinyStories-30M checkpoint instead of yours.

## Learn

```card
type: concept
id: where_params_live
eyebrow: The model
title: Over a third of this model is its vocabulary.
visual:
  kind: bars
  max: 19
  bars:
    - { label: 6 blocks, value: 18.91, text: 18.9M }
    - { label: wte, value: 11.01, text: 11.0M }
    - { label: wpe, value: 0.26, text: 0.26M }
body: |
  6 blocks, 512 wide, 8 heads, 512 tokens of context, and a 21,504-token BPE trained on TinyStories. The token embedding `wte` is a [21,504, 512] table, 11.0M numbers. The output layer reuses that same matrix (**weight tying**), so it costs nothing extra. In GPT-2 small the blocks dominate; in a model this small, the vocabulary is a big share of the budget. Phase 4B comes back to that.
```

```card
type: predict
id: predict_untied
eyebrow: Predict the output
title: Untie lm_head and give it its own matrix. What's the count?
code: |
  n = count_params(config)      # 30,187,520 with tying
  untied = n + 21504 * 512      # lm_head gets its own [21504, 512] weight
  print(f"{untied:,}")
options:
  - text: "41,197,568"
    correct: true
    why: 21,504 × 512 = 11,010,048 more. Untying makes the model 36% bigger without adding a single block.
  - text: "30,187,520"
    why: That's the tied count. An untied lm_head is a second full [vocab, d] matrix.
  - text: "52,207,616"
    why: That adds the vocab matrix twice. wte is already counted; untying adds one more copy.
  - text: "30,209,024"
    why: That adds only a bias of 21,504. The weight matrix is 512 times bigger than that.
hint: 21,504 × 512 is a little over 11 million.
```

```card
type: calc
id: ppl_from_loss
eyebrow: Do the math
title: Val loss is 2.06 nats per token. What's the perplexity?
given:
  - { label: Val loss, value: 2.06 nats per token }
  - { label: Perplexity, value: e^loss }
unit: ppl
answer: { min: 7.7, max: 8.0 }
placeholder: "0.0"
why_ok: e^2.06 ≈ 7.85. The model is as unsure as if it picked uniformly from about 8 tokens at every step, out of 21,504.
why_wrong: Perplexity is e raised to the loss, not the loss times e. e² ≈ 7.39, and a little more for 2.06.
hint: e² is about 7.39. You need e to the 2.06.
```

```card
type: fill_blank
id: six_nd
eyebrow: Fill the blanks
title: Estimate the training compute.
code: |
  tokens = steps * batch_size * block_size
  flops = [[0]] * n_params * [[1]]
tokens: ["6", "2", tokens, steps, n_layer]
answer: ["6", tokens]
tips:
  "2": That's the forward pass alone. The backward pass costs about twice the forward, so training is 6 per parameter per token.
  steps: Each step processes batch_size × block_size tokens. The rule counts tokens, not steps.
  n_layer: The layers are already inside n_params. Counting them again would multiply by 6 twice.
why_ok: "Every token passes every parameter once forward (2 FLOPs: multiply and add) and once backward (about 4). 6 × 30.2M × 245.8M ≈ 4.45 × 10¹⁶."
why_wrong: "The rule is FLOPs per parameter per token. What's that constant, and what counts the work done?"
hint: Forward is 2 FLOPs per parameter per token, and backward costs twice the forward.
```

```card
type: calc
id: dataset_bytes
eyebrow: Do the math
title: How big is the training set on disk as uint16?
given:
  - { label: Train tokens, value: 472.6M }
  - { label: Vocab size, value: "21,504, under uint16's 65,535" }
unit: MB
answer: { min: 940, max: 950 }
placeholder: "0"
why_ok: 472.6M × 2 bytes ≈ 945 MB. Stored as int64, PyTorch's default for indices, it would be 3.8 GB, 4× more, for no benefit.
why_wrong: uint16 is 2 bytes per token. Multiply the token count by 2.
hint: One uint16 is 2 bytes, and a million bytes is 1 MB.
```

```card
type: quiz
ref: val_gap
```

```card
type: quiz
ref: mfu
```

## Build steps

### Count the parameters

```challenge
id: count_params
file: plan.py
minutes: 20
start: { code: "" }
setup:
  - say: Here's the model you're about to train. 6 blocks, 512 wide, 8 heads, 512 tokens of context, and a vocabulary of 21,504 tokens.
    type: "config = dict(vocab_size=21504, block_size=512, n_layer=6, n_head=8, n_embd=512)\n\n"
    pause: 3.0
  - say: "Before you spend GPU time on it, count its parameters from the config alone. Your turn: the embeddings, the blocks and the final LayerNorm."
    type_file: count_params/stub.py
    pause: 1.2
tests: count_params/tests.py
solution: count_params/solution.py
alternatives:
  - count_params/alternatives/with_names.py
  - count_params/alternatives/factored.py
hints:
  logic: |
    embeddings = two tables, d numbers per row
      token table: one row per vocab entry (V)
      position table: one row per position (T)
    blocks = L blocks, each the count from 1.2
    final = the last LayerNorm: gain + bias, d each
    embeddings + blocks + final
      → return the total
    lm_head shares the token table: it adds nothing
    n_head never enters the count
  tools: |
    the two embedding tables
      plain arithmetic: * +   (rows × d, for each table)
    L blocks
      plain arithmetic: * +   (one block's count × L)
    the final LayerNorm
      plain arithmetic: *     (gain and bias, d each)
  nudge: Three groups. The two embedding tables, the stack of blocks, and the LayerNorm at the very end. How big is each table, and what does one block hold?
  concept: "wte is a [vocab_size, d] table and wpe a [block_size, d] table. Each block is 1.2's 12·d² + 13·d, and there are n_layer of them. The final LayerNorm has a gain and a bias. lm_head reuses wte's matrix, so it adds nothing."
  example: |
    Trace it on a tiny config: `V = 10`, `T = 4`, `L = 2`, `d = 3`:

    - `embeddings`: 10 × 3 + 4 × 3 = 42.
    - One block: 12 × 9 + 13 × 3 = 147. Two blocks: `blocks = 294`.
    - `final`: 2 × 3 = 6.
    - Total: 42 + 294 + 6 = `342`.

    Now the real config. If you print the three parts: `embeddings` = 11,272,192, `blocks` = 18,914,304, `final` = 1,024. They add up to 30,187,520.
  skeleton: |
    embeddings = _ * d + _ * d
    blocks = _ * (___ * d * d + ___ * d)
    final = _ * d
  solution: |
    embeddings = V * d + T * d
    blocks = L * (12 * d * d + 13 * d)
    final = 2 * d
on_solve: "30,187,520, the 30.2M that 3.1 and 3.2 quantize, 60.4 MB in fp16. Step 2 is unlocked: how long will training it take?"
mistakes:
  - { file: count_params/mistakes/placeholders.py, fails: returns_number, says: "`embeddings` is still `...`" }
  - { file: count_params/mistakes/untied.py, fails: embeddings, says: "you counted the token embedding and lm_head separately" }
  - { file: count_params/mistakes/untied_bias.py, fails: embeddings, says: "a separate lm_head with a bias" }
  - { file: count_params/mistakes/no_wpe.py, fails: embeddings, says: "the position embedding is missing" }
  - { file: count_params/mistakes/heads.py, fails: blocks, says: "They add no parameters" }
  - { file: count_params/mistakes/one_block.py, fails: blocks, says: "you've counted one block" }
  - { file: count_params/mistakes/matrices_only.py, fails: blocks, says: "only its matrices" }
  - { file: count_params/mistakes/no_final.py, fails: total, says: "the final LayerNorm, ln_f" }
  - { file: count_params/mistakes/final_gain_only.py, fails: total, says: "a gain *and* a bias" }
```

### Estimate the compute

```challenge
id: run_budget
file: plan.py
minutes: 20
start: { from: count_params }
setup:
  - say: Before renting a GPU, estimate the bill. The rule of thumb is that training costs about 6 FLOPs per parameter per token.
    type_file: run_budget/stub.py
    pause: 2.6
  - say: "Your turn: the FLOPs, then the seconds at the speed you really get."
    pause: 1.0
tests: run_budget/tests.py
solution: run_budget/solution.py
alternatives:
  - run_budget/alternatives/named_speed.py
  - run_budget/alternatives/split_passes.py
hints:
  logic: |
    flops = FLOPs per parameter per token
      × n_params × tokens
    peak speed = peak_tflops, in plain FLOPs per second
      (a tera is a trillion)
    real speed = peak speed × mfu
      (you only get a fraction of the peak)
    seconds = flops ÷ real speed
      → return flops and the time in hours
  tools: |
    total training compute
      plain arithmetic: *
    the speed you really get
      1eN                 scientific notation: 1e3 is 1000
      plain arithmetic: *
    time in seconds
      plain arithmetic: /
  nudge: Two lines. FLOPs per parameter per token, times the parameters, times the tokens. Then divide by the FLOPs per second the GPU really delivers.
  concept: "The forward pass costs about 2 FLOPs per parameter per token, a multiply and an add, and the backward pass about twice that: 6·N·D. A GPU rated at peak_tflops does at most peak_tflops × 10¹² FLOPs per second, and real training gets a fraction mfu of that."
  example: |
    Trace it with round numbers: `n_params = 1e6`, `tokens = 1e9`, `peak_tflops = 100`, `mfu = 0.5`:

    - `flops` = 6 × 1e6 × 1e9 = `6e15`.
    - Peak speed: 100 × 1e12 = 1e14 FLOPs per second. Real speed: 1e14 × 0.5 = `5e13`.
    - `seconds` = 6e15 / 5e13 = `120`. The function returns 120 / 3600 ≈ `0.033` hours.

    The real run: `tokens` = 245,760,000 and `n_params` = 30,187,520 give about 4.45e16 FLOPs. At 71e12 × 0.30 = 2.13e13 FLOPs per second that's about 2,090 seconds, or 0.58 hours.
  skeleton: |
    flops = _ * ________ * ______
    seconds = flops / (___________ * 1e__ * ___)
  solution: |
    flops = 6 * n_params * tokens
    seconds = flops / (peak_tflops * 1e12 * mfu)
on_solve: "About 0.58 GPU-hours. The illustrative recording below shows 0.71, and real runs come in above the estimate too: the rule leaves out attention's own FLOPs, evals and data loading, so treat it as a floor. Now the real run."
mistakes:
  - { file: run_budget/mistakes/placeholders.py, fails: flops, says: "`flops` is still `...`" }
  - { file: run_budget/mistakes/forward_only.py, fails: flops, says: "the forward pass alone" }
  - { file: run_budget/mistakes/backward_only.py, fails: flops, says: "the backward pass alone" }
  - { file: run_budget/mistakes/no_mfu.py, fails: hours, says: "as if the GPU ran at 100% of its peak" }
  - { file: run_budget/mistakes/times_mfu.py, fails: hours, says: "You multiplied by mfu" }
  - { file: run_budget/mistakes/no_tera.py, fails: hours, says: "off by a factor of 10¹²" }
  - { file: run_budget/mistakes/already_hours.py, fails: hours, says: "`seconds` already holds hours" }
```

### Train on TinyStories

```replay
id: tinystories_30m
intro: "Tokenize TinyStories, then train for 15,000 steps with 1.3's schedule and clipping. It takes about 45 minutes on an RTX 3090, a few hours on a Colab T4. Watch a recording first so you know what healthy looks like."
log: replays/tinystories-30m.jsonl
speed: 40
recorded:
  hardware: RTX 3090
  illustrative: true
  note: Sample data written for the course, consistent with the 30M baseline used in 3.1 and 3.2. Not a real recording. Replace it with a log of your own run.
chart:
  title: Validation perplexity every 1,000 steps
  flag_above: 10
  x_labels: [step 1000, step 15000]
  caption: Red bars are above the pass limit of 10. The run crosses it a little before step 5,000 and keeps improving to 7.84, still falling slowly at the end.
summary:
  - { label: Val perplexity, value: "7.84" }
  - { label: GPU time, value: 0.71 h }
  - { label: Tokens seen, value: 245.8M (0.52 epoch) }
run_yourself: |
  # on your machine or Colab, from the course repo
  python prepare.py --dataset tinystories --tokenizer tok/tinystories-21k.json
  python train_gpt.py --config tinystories_30m
  python sample.py --ckpt runs/tinystories-30m/best.pt --prompt "Once upon a time"
```

### Record your metrics

```metrics
id: record
intro: Your val perplexity and the run's wall-clock GPU hours. They're compared with the reference run and the pass limit.
```

## Hints

- Before the long run, overfit one batch: 200 steps on a single batch should drive the loss under 0.5. If it can't, the bug is in the model, not the data.
- The loss at step 0 should be close to ln(vocab_size), 9.98 here. Much higher means the output layer starts too confident; scale its init down.
- Keep `train.bin` as uint16 and load it with `np.memmap`, so the 945 MB never has to fit in RAM at once.
- On a T4, drop bf16 for fp16 with a GradScaler, or stay in fp32 and halve the batch. The T4 has no bf16 tensor cores.

## Done when

- [ ] count_params gives 30,187,520 for the 30M config {#count_params auto=challenge:count_params}
- [ ] run_budget estimates the run before you start it {#run_budget auto=challenge:run_budget}
- [ ] Your 30M model writes a readable story from "Once upon a time" {#stories}
- [ ] Val perplexity and GPU hours recorded {#metrics_recorded auto=metrics}
