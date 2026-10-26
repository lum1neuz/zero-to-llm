---
id: "3.5"
title: AWQ and NF4
summary: Two more ways to spend 4 bits well. AWQ scales up the weights that big activations depend on; NF4 puts its 16 levels where normally distributed weights actually fall.
phase: "3"
hours: 5
type: run
status: ready
protected: true
cut: null
prerequisites:
  - "3.4"
papers:
  - title: AWQ
    authors: Lin et al.
    year: 2023
    url: https://arxiv.org/abs/2306.00978
  - title: QLoRA
    authors: Dettmers et al.
    year: 2023
    url: https://arxiv.org/abs/2305.14314
quality_metric:
  id: ppl_delta
  label: Best int4 perplexity change vs fp16
  unit: "%"
  note: The better of AWQ int4 g128 and NF4, against the fp16 baseline from 3.4. GPTQ was +4.5% in the reference run.
  inputs:
    - { id: ppl_fp16, label: fp16 val perplexity, hint: Your baseline from 3.4 }
    - { id: ppl_best, label: Best 4-bit perplexity, hint: "AWQ or NF4, whichever is lower" }
  compute: pct_change
  reference: 5.0
  pass: { max: 7.0 }
systems_metric:
  id: weight_gb
  label: Llama 3.1 8B weight memory at 4 bits
  unit: GB
  note: NF4 with double quantization, as loaded. Embeddings and lm_head stay in bf16.
  inputs:
    - { id: weight_gb, label: Weight memory (GB), hint: "model.get_memory_footprint() / 1e9" }
  compute: value
  reference: 5.7
  pass: { max: 6.5 }
quiz:
  - id: awq_calibration
    q: GPTQ uses calibration data to build a Hessian and correct every column. What does AWQ use it for?
    answer: 1
    options:
      - text: The same Hessian, with a faster solver
        why: AWQ never builds H and never updates weights column by column. That's why it's simpler and quicker to run.
      - text: The average |x| of each input channel, to pick the scaling factors
        why: That's all of it. One number per channel, raised to alpha, plus a small grid search over alpha per layer. Because it fits so little, it's also less prone to overfitting the calibration set.
      - text: To pick which 1% of weights to keep in fp16
        why: AWQ's paper starts from that observation, but mixed precision is awkward for kernels. Scaling the salient channels gets most of the benefit with every weight still in int4.
  - id: why_quantiles
    q: Why does NF4 put its 16 levels at quantiles of a normal distribution instead of spacing them evenly like int4?
    answer: 0
    options:
      - text: Weights are roughly normal, so each level then gets an equal share of them
        why: Most weights sit near zero and few near the block's max. Quantile levels crowd together near zero, where the weights are, so no code is wasted on empty range.
      - text: It makes the levels exactly representable in fp16
        why: The levels are looked up from a 16-entry table, so they can be any value. Representability doesn't come into it.
      - text: It lets the matmul run in integer arithmetic
        why: It's the opposite. NF4 levels aren't evenly spaced, so you can't multiply the codes as integers. Kernels look each code up and compute in bf16.
  - id: dq_saving
    q: Double quantization saves 0.373 bits per weight. About how much is that on the 7 billion weights NF4 quantizes in Llama 3.1 8B?
    answer: 1
    options:
      - text: About 33 MB
        why: "Check the units: 7 × 10⁹ × 0.373 bits is 2.6 × 10⁹ bits, and dividing by 8 gives bytes."
      - text: About 330 MB
        why: 7 × 10⁹ × 0.373 / 8 ≈ 326 MB. Small next to 3.6 GB of weights, but on a 12 GB card it's room for a longer context.
      - text: About 3.3 GB
        why: That would be 3.7 bits per weight, almost the whole weight. The saving is only on the block constants.
---

## Goal

GPTQ corrects errors after rounding. **AWQ** and **NF4** make the rounding itself land better. AWQ (activation-aware weight quantization) notices that a few input channels carry much larger activations, so their weights matter more, and scales those weight columns up before quantizing so they get finer steps. NF4 replaces int4's evenly spaced levels with 16 levels placed at the quantiles of a normal distribution, where weights actually fall, and QLoRA adds **double quantization** to shrink the per-block constants. Write both on small tensors, then run AWQ and NF4 on Llama 3.1 8B and compare them with GPTQ.

## Learn

```card
type: concept
id: salient_channels
eyebrow: AWQ's idea
title: Big activations make their weights matter more.
visual:
  kind: bars
  max: 0.021
  bars:
    - { label: "α = 0 (RTN)", value: 0.0201, text: "0.020" }
    - { label: "α = 0.25", value: 0.0140, text: "0.014" }
    - { label: "α = 0.5", value: 0.0047, text: "0.0047" }
body: |
  An error on weight w_ij reaches the output multiplied by x_j. If channel j's activations are 6× bigger, so is the damage. AWQ multiplies column j of W by s_j = mean|x_j|^α before quantizing, which gives those weights finer steps relative to their row, and divides x_j by s_j at run time so the product doesn't change. The bars show output error on this lesson's layer for three values of α.
```

```card
type: predict
id: predict_nf4
eyebrow: Predict the output
title: A weight normalized to 0.30. Which value does NF4 store for it?
code: |
  # the eight positive NF4 levels
  NF4_POS = [0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
  w = 0.30                   # already divided by its block's absmax
options:
  - text: "0.3379"
    correct: true
    why: 0.30 sits between 0.2461 and 0.3379. It's 0.054 from the first and 0.038 from the second, so it snaps to 0.3379, code 11.
  - text: "0.2461"
    why: That's the level below. Measure both gaps, 0.054 below and 0.038 above. The nearest level wins.
  - text: "0.2857"
    why: That's int4 absmax, 2 / 7. int4's levels are evenly spaced at multiples of 1/7. NF4's come from a table.
  - text: "0.30"
    why: NF4 can only store one of its 16 levels. Every weight snaps to the nearest one.
hint: Find the two levels that 0.30 sits between, then check which one is closer.
```

```card
type: fill_blank
id: nearest_level
eyebrow: Fill the blanks
title: Find each weight's nearest NF4 level.
code: |
  wn = wb / absmax                                  # [blocks, block], all in [-1, 1]
  idx = (wn[..., None] - NF4).[[0]]().[[1]](dim=-1)  # [blocks, block], 0…15
tokens: [abs, argmin, argmax, round, min, sum]
answer: [abs, argmin]
tips:
  argmax: That picks the level *furthest* from the weight. You want the closest.
  round: Rounding works for evenly spaced levels like int4. NF4's levels are uneven, so you have to measure the distance to each one.
  min: "`min(dim=…)` gives the distance itself, and its `.values` aren't an index. You want which level, not how far."
  sum: Summing the distances mixes all 16 levels into one number. Pick the single closest instead.
why_ok: "`wn[..., None] - NF4` is a [blocks, block, 16] table of signed gaps. `abs` makes them distances, and `argmin` over the last axis picks the closest level's index, 0 to 15."
why_wrong: First turn the signed gaps into distances, then take the *index* of the smallest one.
hint: "Two steps: make every gap positive, then ask which of the 16 is smallest. You want the position, not the value."
```

```card
type: calc
id: dq_bits
eyebrow: Do the math
title: With double quantization, how many bits per weight do the block constants cost?
given:
  - { label: Block size, value: 64 weights }
  - { label: Each block's absmax, value: 8-bit float }
  - { label: Second-level constant, value: one fp32 per 256 blocks }
unit: bits
answer: { min: 0.126, max: 0.128 }
placeholder: "0.000"
why_ok: "8 / 64 + 32 / (64 × 256) = 0.125 + 0.002 = 0.127 bits. Without double quantization each block's absmax is fp32: 32 / 64 = 0.5 bits. So NF4 costs 4.127 bits per weight instead of 4.5."
why_wrong: "Each 8-bit absmax is shared by 64 weights. Each 32-bit second-level constant is shared by 256 blocks of 64 weights. Add the two shares."
hint: Spread each constant over every weight that shares it. The 8-bit one covers 64 weights; the 32-bit one covers 64 × 256.
```

```card
type: calc
id: nf4_gb
eyebrow: Do the math
title: Llama 3.1 8B in NF4. How many GB of weights?
given:
  - { label: Weights NF4 quantizes, value: 6.98 × 10⁹ at 4.127 bits }
  - { label: Embeddings and lm_head, value: 1.05 × 10⁹ in bf16 (2 bytes) }
unit: GB
answer: { min: 5.6, max: 5.8 }
placeholder: "0.0"
why_ok: 6.98 × 10⁹ × 4.127 / 8 = 3.60 GB, plus 1.05 × 10⁹ × 2 = 2.10 GB, so 5.70 GB. More than a third of it is the two vocabulary matrices, because Llama 3's 128k vocabulary makes them big.
why_wrong: Do the two parts separately. The quantized weights cost 4.127 bits each, divided by 8 for bytes. The embeddings and lm_head cost 2 bytes each.
hint: Bits to bytes is ÷ 8, and 10⁹ bytes is 1 GB.
```

```card
type: quiz
ref: awq_calibration
```

```card
type: quiz
ref: why_quantiles
```

```card
type: quiz
ref: dq_saving
```

## Build steps

### Activation-aware scaling

```challenge
id: awq_scale
file: awq.py
minutes: 35
start: { code: "import torch\n\n" }
setup:
  - say: "32 calibration tokens with 8 features. Features 2 and 6 are salient: their activations run 6× and 3× bigger than the rest."
    type: "g = torch.Generator().manual_seed(1)\nX = torch.randn(32, 8, generator=g) * torch.tensor([1.0, 0.8, 6.0, 1.2, 0.9, 1.0, 3.0, 0.7])\nW = 0.3 * torch.randn(4, 8, generator=g)\n\n"
    pause: 3.0
  - say: "int4 with one scale per row, from 3.2, returning the dequantized weights straight away."
    type: "def quantize_rows_int4(W):\n    scale = W.abs().amax(dim=1, keepdim=True) / 7\n    return torch.round(W / scale).clamp(-7, 7) * scale\n"
    pause: 2.0
  - say: "AWQ scales each weight column by its channel's average activation to the power alpha, quantizes, and scales back. The loop tries five alphas; alpha 0 is plain round-to-nearest. Your turn: fill in s and the return."
    type_file: awq_scale/stub.py
    pause: 1.2
tests: awq_scale/tests.py
solution: awq_scale/solution.py
alternatives:
  - awq_scale/alternatives/pow_method.py
  - awq_scale/alternatives/explicit_steps.py
hints:
  start:
    try: |
      print(X[:, 0].abs().mean())
      print(X[:, 2].abs().mean())
    see: |
      tensor(0.7271)
      tensor(3.4993)
    next: "That's the average activation size of channel 0 and of the salient channel 2. `s` is that average to the power `alpha`, one number per channel, so its shape is [8]. Get the average for all eight channels at once, then stretch the weight columns by `s`, quantize, and shrink them back."
  logic: |
    for each input channel:
      average size of its activations over the tokens
        → s = that average ^ alpha   (alpha 0 → s = 1)
    stretch each weight column by its s
      → quantize every row to int4
    shrink each column back by its s
      → return the weights
  tools: |
    average size per channel
      t.abs()          each value without its sign
      t.mean(dim)      the average along dim
    raise to alpha
      t ** x           every value to the power x
    stretch, quantize, shrink
      plain arithmetic: * /   (a [channels] vector applies to
                               every row)
      quantize_rows_int4(t)   the helper typed above
  nudge: "Two moves. Measure how big each input channel's activations are, and use that to stretch the weight columns before quantizing. What has to happen afterwards so X @ W_hat.T still means the same thing?"
  concept: "`X.abs().mean(dim=0)` is the average |x| per input channel, shape [8]. Raise it to `alpha` for s. W is [out, in], so `W * s` stretches each column j by s_j. Quantize that with `quantize_rows_int4`, then divide by s to bring the weights back to their own scale. In a real model the 1 / s is folded into the previous layer instead."
  example: |
    Trace it on 2 tokens and 3 channels, with one row of weights: `X = [[1, 4, 0.5], [-2, -6, 1]]`, `W = [[0.7, 0.11, 0.3]]`, `alpha = 0.5`.

    - Mean |x| per channel: `[1.5, 5.0, 0.75]`. Channel 1 is the salient one.
    - `s = √[1.5, 5.0, 0.75] ≈ [1.225, 2.236, 0.866]`.
    - Stretched: `W * s ≈ [0.857, 0.246, 0.260]`. The row's largest is now 0.857, so one int4 step is 0.857 / 7 ≈ 0.122, and the codes are `[7, 2, 2]`.
    - Back on the grid and divided by `s`: `≈ [0.700, 0.110, 0.283]`.
    - With `alpha = 0`, the step is 0.1 and the middle weight 0.11 rounds to 0.1, a 0.01 error on the channel with the biggest activations. With `alpha = 0.5` that weight comes back as 0.1095, an error of about 0.0005, and the output error drops from about 0.0026 to 0.00016.
  skeleton: |
    s = X.___().____(dim=___) ** _____
    return quantize_rows_int4(W _ s) _ s
  solution: |
    s = X.abs().mean(dim=0) ** alpha
    return quantize_rows_int4(W * s) / s
on_solve: "Alpha 0.5 cuts output error more than 4×, and the higher alphas overshoot: stretch the salient columns too far and they set each row's scale, which blurs every other weight in the row. That trade-off is why AWQ searches alpha separately for every layer instead of fixing it."
mistakes:
  - { file: awq_scale/mistakes/placeholders.py, fails: shape, says: "still returns `...`" }
  - { file: awq_scale/mistakes/s_placeholder.py, fails: shape, says: "`s` is still `...`" }
  - { file: awq_scale/mistakes/per_token.py, fails: shape, says: "one per token" }
  - { file: awq_scale/mistakes/no_unscale.py, fails: activation_aware, says: "never divided back" }
  - { file: awq_scale/mistakes/inverted.py, fails: activation_aware, says: "shrank the salient columns" }
  - { file: awq_scale/mistakes/weights_mean.py, fails: activation_aware, says: "from the weights" }
  - { file: awq_scale/mistakes/max_not_mean.py, fails: activation_aware, says: "the largest |x|" }
  - { file: awq_scale/mistakes/fixed_alpha.py, fails: activation_aware, says: "Use the `alpha` argument" }
  - { file: awq_scale/mistakes/no_abs.py, fails: activation_aware, says: "nan" }
```

### NF4 and its codebook

```challenge
id: nf4
file: nf4.py
minutes: 35
start: { code: "import torch\n\n" }
setup:
  - say: "512 weights drawn from a normal distribution, like the weights of a trained Linear."
    type: "w = 0.05 * torch.randn(512, generator=torch.Generator().manual_seed(0))\n\n"
    pause: 2.0
  - say: "NF4's 16 levels, the table bitsandbytes uses. They're quantiles of a normal distribution, scaled to [−1, 1], with an exact zero. Look how they crowd together near 0."
    type: "NF4 = torch.tensor([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,\n                    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0])\n\n"
    pause: 3.2
  - say: "The baseline: 3.2's int4 absmax on blocks, dequantized straight away."
    type: "def int4_blocks(w, block):\n    wb = w.reshape(-1, block)\n    scale = wb.abs().amax(dim=1, keepdim=True) / 7\n    return (torch.round(wb / scale).clamp(-7, 7) * scale).reshape(-1)\n"
    pause: 2.0
  - say: "NF4 works on blocks too, but instead of rounding, each weight looks up its nearest level in the table. Your turn: fill in wn, idx and the dequantize line."
    type_file: nf4/stub.py
    pause: 1.2
tests: nf4/tests.py
solution: nf4/solution.py
alternatives:
  - nf4/alternatives/unsqueeze.py
  - nf4/alternatives/min_indices.py
  - nf4/alternatives/int_index.py
hints:
  start:
    try: |
      wb = w.reshape(-1, 64)
      print(wb.shape, wb[0].abs().max())
      print(wb[0, 0] / wb[0].abs().max())
    see: |
      torch.Size([8, 64]) tensor(0.1163)
      tensor(0.0541)
    next: "That's 8 blocks of 64, the first block's absmax, and its first weight squeezed into −1…1. `wn` does that for every weight with its own block's absmax, which has to keep a column shape. Then ask which of the 16 NF4 levels a squeezed value is nearest to."
  logic: |
    quantize, for each block:
      divide the block by its absmax → values in −1…1
      for each weight:
        distance to each of the 16 levels
          → code = position of the closest one (0…15)
      return codes and absmax
    dequantize:
      level for each code → times its block's absmax
        → flatten to one row
  tools: |
    squeeze each block into -1…1
      t / x            divides every value by x
    distance to each of the 16 levels
      t[..., None]     adds a new last axis of size 1
      a - b            subtracts, broadcasting over the new axis
      t.abs()          each value without its sign
    position of the closest level
      t.argmin(dim)    the index of the smallest value along dim
    look the levels up again
      table[idx]       one table entry per code in idx
      t.long()         converts to a normal integer type
    undo the scaling and flatten
      t * x            multiplies every value by x
      t.reshape(-1)    flattens to one row
  nudge: "Three moves. Squeeze each block into [−1, 1], find each weight's closest level in the table, and on the way back look the levels up again. Which tensor goes inside the square brackets?"
  concept: "`wb / absmax` puts every block in [−1, 1], since absmax is a [blocks, 1] column. `wn[..., None] - NF4` broadcasts a new last axis against the 16 levels, giving a [blocks, block, 16] table of gaps; `.abs().argmin(dim=-1)` picks the closest. On the way back, `NF4[idx.long()]` looks the levels up (a uint8 index would be read as a mask), times `absmax`, then `reshape(-1)`."
  example: |
    Trace it with `block = 4` on 8 weights, `w = [0.10, -0.04, 0.02, -0.20, 0.03, 0.01, -0.05, 0.06]`:

    - Two blocks. `absmax = [[0.20], [0.06]]`.
    - `wn = [[0.5, -0.2, 0.1, -1.0], [0.5, 0.167, -0.833, 1.0]]`.
    - The table is [2, 4, 16]. For 0.5 the closest levels are 0.4407 (gap 0.059) and 0.5626 (gap 0.063), so it takes the first, code 12.
    - `idx = [[12, 5, 8, 0], [12, 9, 1, 15]]`, shape [2, 4].
    - Levels: `[[0.4407, -0.1848, 0.0796, -1.0], [0.4407, 0.1609, -0.6962, 1.0]]`.
    - Times `absmax` and flattened: `[0.0881, -0.0370, 0.0159, -0.2000, 0.0264, 0.0097, -0.0418, 0.0600]`. Shape [8], close to `w`.
  skeleton: |
    wn = wb / ______
    idx = (wn[..., ____] - NF4).___().______(dim=___)
    return (NF4[idx.____()] * ______).reshape(___)
  solution: |
    wn = wb / absmax
    idx = (wn[..., None] - NF4).abs().argmin(dim=-1)
    return (NF4[idx.long()] * absmax).reshape(-1)
on_solve: "NF4's error is about 30% lower than int4 absmax at the same block size, with the same 4 bits per weight. The win comes from putting levels where weights are dense, and from the 16th level that symmetric int4 leaves unused. On a GPU, bitsandbytes stores two codes per byte and dequantizes to bf16 inside the matmul kernel."
mistakes:
  - { file: nf4/mistakes/placeholders.py, fails: codes, says: "is still `...`, the placeholder" }
  - { file: nf4/mistakes/idx_placeholder.py, fails: codes, says: "`idx` is still `...`" }
  - { file: nf4/mistakes/no_new_axis.py, fails: codes, says: "new last axis" }
  - { file: nf4/mistakes/wrong_dim.py, fails: codes, says: "one code per weight" }
  - { file: nf4/mistakes/no_normalize.py, fails: nearest_level, says: "never divided by its block's absmax" }
  - { file: nf4/mistakes/signed_gap.py, fails: nearest_level, says: "signed gaps" }
  - { file: nf4/mistakes/argmax.py, fails: nearest_level, says: "furthest" }
  - { file: nf4/mistakes/round_like_int4.py, fails: nearest_level, says: "evenly spaced" }
  - { file: nf4/mistakes/return_placeholder.py, fails: round_trip, says: "still returns `...`" }
  - { file: nf4/mistakes/uint8_index.py, fails: round_trip, says: "`.long()`" }
  - { file: nf4/mistakes/no_absmax.py, fails: round_trip, says: "between −1 and 1" }
  - { file: nf4/mistakes/no_flatten.py, fails: round_trip, says: "one row per block" }
```

### AWQ and NF4 on Llama 3.1 8B

```replay
id: awq_nf4_8b
intro: "Two runs on the same 12 GB card. AWQ searches alpha block by block (about 15 minutes); NF4 needs no calibration and quantizes as it loads. Then the same perplexity eval as 3.4. Watch a recording first."
log: replays/awq-nf4-llama-8b.jsonl
speed: 40
recorded:
  hardware: RTX 3060 12 GB
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: Best alpha per block
  flag_above: 0.6
  x_labels: [blocks.0, blocks.31]
  caption: "Most blocks settle between 0.3 and 0.55. A few late blocks, where activations are most uneven, want more of the scale moved into the weights."
summary:
  - { label: AWQ int4 g128, value: "6.24 → 6.55 (+5.0%)" }
  - { label: NF4 + double quant, value: "6.24 → 6.63 (+6.3%)" }
  - { label: GPTQ (3.4), value: "6.24 → 6.52 (+4.5%)" }
  - { label: NF4 weight memory, value: 5.70 GB }
run_yourself: |
  # on your machine or Colab, from the course repo
  python awq_quantize.py --model meta-llama/Llama-3.1-8B \
    --bits 4 --group-size 128 --calib pile --nsamples 128
  python nf4_load.py --model meta-llama/Llama-3.1-8B --double-quant --report-memory
  python eval_ppl.py --models awq-g128,nf4-dq --eval-samples 2000
```

### Record your metrics

```metrics
id: record
intro: The better of AWQ and NF4 against fp16, and the NF4 model's weight memory as loaded.
```

## Hints

- AWQ's real implementation normalizes s by √(max(s) · min(s)) so the numbers stay moderate. With absmax quantization that doesn't change the result, since multiplying every column by the same constant scales the row's absmax too.
- AWQ folds the 1 / s into the previous operation (a LayerNorm's weight, or the previous Linear), so inference pays nothing for it.
- NF4 in bitsandbytes: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`. Weights stay NF4 in memory and are dequantized to bf16 on the fly, so NF4 saves memory but isn't faster than bf16.
- NF4 is the format QLoRA trains adapters on top of. You'll load exactly this model again in 4A.2.

## Done when

- [ ] AWQ beats round-to-nearest on the lesson's layer {#awq auto=challenge:awq_scale}
- [ ] NF4 beats int4 absmax on normally distributed weights {#nf4 auto=challenge:nf4}
- [ ] Best 4-bit perplexity change and NF4 memory recorded {#metrics_recorded auto=metrics}
