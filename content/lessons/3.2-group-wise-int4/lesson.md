---
id: "3.2"
title: Group-wise int4
summary: Give every 128 weights their own scale, take your 30M model's Linear layers down to 4 bits, and see why one scale per row isn't enough at int4.
phase: "3"
hours: 6
type: build
status: ready
protected: true
cut: null
prerequisites:
  - "3.1"
papers:
  - title: "The case for 4-bit precision: k-bit Inference Scaling Laws"
    authors: Dettmers & Zettlemoyer
    year: 2022
    url: https://arxiv.org/abs/2212.09720
  - title: "ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers"
    authors: Yao et al.
    year: 2022
    url: https://arxiv.org/abs/2206.01861
quality_metric:
  id: ppl_delta
  label: Perplexity change vs fp16
  unit: "%"
  inputs:
    - { id: ppl_fp16, label: fp16 val perplexity, hint: From your baseline eval }
    - { id: ppl_int4, label: int4 g128 val perplexity, hint: "Same eval, quantized" }
  compute: pct_change
  reference: 3.6
  pass: { max: 5.0 }
systems_metric:
  id: weight_mb
  label: int4 weight memory
  unit: MB
  note: The 24 nn.Linear weights packed two per byte, plus one fp16 scale per group of 128. In fp16 they were 37.7 MB, in int8 18.9 MB. Embeddings stay fp16, so the whole model goes from 60.4 MB to about 32.4 MB.
  inputs:
    - { id: weight_mb, label: int4 weight memory (MB), hint: Packed Linear weights + group scales }
  compute: value
  reference: 9.7
  pass: { max: 10.0 }
quiz:
  - id: outlier_group
    q: A row of 4096 weights has one weight at 40.0, and every other weight is under 1.0. With int4 and groups of 128, which weights lose precision to the outlier?
    answer: 0
    options:
      - text: Only the 127 others in its group
        why: Each group has its own scale. The outlier sets the scale for its own group of 128, and the other 31 groups in the row never see it.
      - text: All 4095 others in the row, same as per-row
        why: That's 3.1's per-row scheme. With groups, each block of 128 picks its own scale, so the damage stops at the group boundary.
      - text: None. int4 clips 40.0 to 7 and the rest are fine
        why: Absmax maps 40.0 to exactly 7, so nothing clips. The cost is the scale of 40 / 7 ≈ 5.7, which rounds the outlier's small neighbours to 0.
  - id: smaller_groups
    q: You shrink the group size from 128 to 32. What changes?
    answer: 1
    options:
      - text: Error drops and memory stays the same
        why: There are 4× as many groups, so 4× as many fp16 scales. The scale overhead goes from 0.125 to 0.5 bits per weight.
      - text: Error drops, and scales grow from 0.125 to 0.5 bits per weight
        why: Fewer weights share each scale, so each group's max sits closer to its typical weights. Each fp16 scale (16 bits) is now spread over 32 weights instead of 128.
      - text: Error grows, because each scale sees fewer weights
        why: It's the reverse. A smaller group's largest magnitude is closer to the rest of the group, so the int4 steps get finer.
  - id: why_7
    q: Signed int4 holds −8…7. Why does absmax map the largest |w| to ±7 and leave −8 unused?
    answer: 0
    options:
      - text: So +amax and −amax both land on a value int4 can hold
        why: Same reason as ±127 in 3.1. Dividing by 7 keeps the grid symmetric, and +7 exists in int4 while +8 doesn't.
      - text: −8 is reserved to mark padding
        why: Nothing reserves it. It just goes unused, which costs 1 of 16 levels.
      - text: It doesn't matter. Dividing by 8 is just as accurate
        why: With amax / 8, a positive amax lands on +8, which int4 can't hold. Clamping it to 7 costs a whole step on the biggest weight, the one that matters most.
---

## Goal

Quantize to **int4** with **one absmax scale per group** of weights instead of one per row. At int4 there are only 15 levels, so a single large weight wrecks a whole row. Split the row into groups of `g`, give each group its own scale, and compare the error against per-row int4. Then quantize your TinyStories 30M model with groups of 128 and measure what it costs and saves.

## Learn

```card
type: concept
id: why_groups
eyebrow: Why groups
title: At int4, one big weight blurs its whole row.
visual:
  kind: bars
  max: 0.09
  bars:
    - { label: row, value: 0.0864, text: "0.086" }
    - { label: g = 8, value: 0.0466, text: "0.047" }
    - { label: g = 4, value: 0.0257, text: "0.026" }
body: |
  int4 has 15 levels, −7…7, instead of int8's 255. Take a row of 16 weights with one at 2.40: per-row absmax sets the step to 2.40 / 7 ≈ 0.34, so the other 15 weights all squeeze onto −1, 0 and 1. Give every **group** of `g` neighbours its own scale and the outlier only blurs its own group. The bars show mean round-trip error on that row.
```

```card
type: fill_blank
id: group_scale
eyebrow: Fill the blanks
title: Complete group-wise absmax.
code: |
  def quantize_groups(w, g):
      wg = w.reshape([[0]])
      scale = wg.abs().amax(dim=1, [[1]]) / [[2]]
      q = torch.round(wg / scale).to(torch.int8)
      return q, scale
tokens: ["g, -1", "-1, g", keepdim=False, keepdim=True, "8", "7"]
answer: ["-1, g", keepdim=True, "7"]
tips:
  "g, -1": That makes `g` rows. Each group should be one row of `g` neighbours, so `g` goes last.
  keepdim=False: Without it, scale has shape `[groups]` and lines up with the columns of `wg`. `keepdim=True` keeps a `[groups, 1]` column, one scale per row.
  "8": int4 holds −8…7. Dividing by 8 would push +amax to +8, which doesn't fit.
why_ok: "Each row of `wg` is one group, `scale` is a `[groups, 1]` column, and each group's largest |w| lands on exactly ±7."
why_wrong: "Check each blank: groups are rows of length `g`, the scale must stay a column, and the int4 limit is the largest value int4 can hold."
hint: "`-1` in a reshape means \"work this one out\". You want each row to be one group of `g` weights."
```

```card
type: predict
id: predict_groups
eyebrow: Predict the output
title: Two groups of two. What prints?
code: |
  wg = torch.tensor([[1.0, -4.0], [0.4, 1.0]])
  scale = wg.abs().amax(dim=1, keepdim=True) / 7
  q = torch.round(wg / scale)
  print(q.tolist())
options:
  - text: "[[2.0, -7.0], [1.0, 2.0]]"
    why: That's one scale (4 / 7) for both groups, per-row style. The second group has its own amax of 1.0, so its scale is 1 / 7.
  - text: "[[2.0, -7.0], [3.0, 7.0]]"
    correct: true
    why: "Scales are 4/7 and 1/7. Row 1: 1.0 → 1.75 → 2 and −4.0 → −7. Row 2: 0.4 → 2.8 → 3 and 1.0 → 7. Each group's biggest magnitude lands on ±7."
  - text: "[[2.0, -28.0], [1.0, 7.0]]"
    why: That's what you'd get without `keepdim=True`. A `[2]`-shaped scale lines up with the columns, so −4.0 gets divided by 1/7.
  - text: "[[1.0, -2.0], [0.0, 0.0]]"
    why: That multiplies by the scale. Dividing by it is what stretches each group across −7…7.
hint: Work out each row's scale separately. Row 1's largest magnitude is 4.0, row 2's is 1.0.
```

```card
type: reorder
id: group_round_trip
eyebrow: Put it in order
title: Arrange the group-wise round trip.
lines:
  - wg = w.reshape(-1, g)
  - scale = wg.abs().amax(dim=1, keepdim=True) / 7
  - q = torch.round(wg / scale).clamp(-7, 7).to(torch.int8)
  - w_hat = (q.float() * scale).reshape(-1)
  - err = (w - w_hat).abs().max()
shuffle: [2, 4, 0, 3, 1]
why_ok: "Group, scale, quantize, dequantize, compare. The reshape comes first so each group is a row, and `w_hat` flattens back at the end so it lines up with `w`."
why_wrong: "Something runs before its input exists. Follow the names: `scale` needs `wg`, `q` needs `scale`, `w_hat` needs `q`."
hint: Follow the variable names. Each line can only use names that an earlier line created.
```

```card
type: calc
id: bits_per_weight
eyebrow: Do the math
title: With groups of 128 and fp16 scales, how many bits does each weight really cost?
given:
  - { label: Bits per int4 weight, value: "4" }
  - { label: Group size, value: "128" }
  - { label: One scale per group, value: fp16 (16 bits) }
unit: bits
answer: { min: 4.12, max: 4.13 }
placeholder: "0.0"
why_ok: 4 + 16 / 128 = 4.125 bits. For Llama 3.1 8B that's about 4.1 GB instead of 4.0. With groups of 32 it's 4.5 bits.
why_wrong: Each weight costs its 4 bits plus its share of the group's scale. One 16-bit scale is shared by 128 weights.
hint: Spread the 16-bit scale evenly over the 128 weights that share it, then add the 4 bits each weight has itself.
```

```card
type: quiz
ref: outlier_group
```

```card
type: quiz
ref: smaller_groups
```

```card
type: quiz
ref: why_7
```

## Build steps

### Write quantize_groups

```challenge
id: quantize_groups
file: groupwise.py
minutes: 40
start: { code: "import torch\n\n" }
setup:
  - say: Here's a row of 16 weights. One of them, 2.40, is far bigger than the rest.
    type: "w = torch.tensor([0.12, -0.31, 0.05, 0.22, -0.18, 0.09, 2.40, -0.27, 0.15, -0.08, 0.30, -0.11, -0.25, 0.19, -0.04, 0.28])\n\n"
    pause: 2.6
  - say: "In 3.1 one scale covered the whole row. This time every g neighbours get their own: lay the row out as groups, find each group's biggest magnitude, and map it to int4's edge."
    pause: 4.0
  - say: "Here's the stub. Your turn: fill in wg, scale and q, then run the tests."
    type_file: quantize_groups/stub.py
    pause: 1.2
tests: quantize_groups/tests.py
solution: quantize_groups/solution.py
alternatives:
  - quantize_groups/alternatives/view.py
  - quantize_groups/alternatives/max_values.py
  - quantize_groups/alternatives/explicit_rows.py
  - quantize_groups/alternatives/no_clamp.py
hints:
  logic: |
    wg = w laid out as rows of g weights   (one group per row)
    for each group (row of wg):
      find its largest size (ignore the sign)
        → scale = that size ÷ 7   (one scale per group, a column)
      divide the row by its scale → round to a whole number
      outside −7…7?
        yes → pull it back to the edge
        no  → keep it
    store the whole numbers in one byte each
      → return q and scale
  tools: |
    lay the row out as groups
      t.reshape(a, b)    same values as a x b; -1 = work it out
    find each group's largest size
      t.abs()            each value without its sign
      t.amax(dim, keepdim=True)   biggest along dim, as a column
    scale, then whole numbers
      t / x              divides every value by x
      torch.round(t)     nearest whole number
      t.clamp(lo, hi)    squeezes values into [lo, hi]
    store in one byte each
      t.to(dtype)        converts; torch.int8 is 1 byte
  nudge: Three moves. Lay the row out so each group is one row, find each row's largest magnitude, and map it to the edge of int4. Which value is that edge?
  concept: "`w.reshape(-1, g)` turns 16 weights into a `[4, 4]` table with one group per row (`-1` means \"work it out\"). `amax(dim=1, keepdim=True)` takes the max across each row and keeps a `[4, 1]` column, so `wg / scale` divides each row by its own scale. int4 holds −8…7. Map the largest |w| to ±7 so the grid stays symmetric, like ±127 in 3.1."
  example: |
    Trace it on 8 weights with `g = 4`, `w = [0.4, -1.0, 0.25, 0.1, 4.0, -1.9, 1.0, 0.2]`:

    - `wg` is `[2, 4]`: row 0 is `[0.4, -1.0, 0.25, 0.1]`, row 1 is `[4.0, -1.9, 1.0, 0.2]`.
    - Largest magnitudes, as a `[2, 1]` column: `[[1.0], [4.0]]`.
    - `scale`: `[[0.1429], [0.5714]]`. That's 1.0 / 7 and 4.0 / 7.
    - Row 0 divided by its scale: `[2.8, -7.0, 1.75, 0.7]`, which rounds to `[3, -7, 2, 1]`.
    - Row 1 divided by its scale: `[7.0, -3.325, 1.75, 0.35]`, which rounds to `[7, -3, 2, 0]`.

    Each group's biggest weight lands on ±7. The `4.0` in row 1 doesn't squash row 0, because the two groups have separate scales.
  skeleton: |
    wg = w.______(-1, ___)
    scale = wg.abs().____(dim=___, keepdim=____) / ___
    q = torch.round(wg / scale).clamp(____, ____).to(torch.int8)
  solution: |
    wg = w.reshape(-1, g)
    scale = wg.abs().amax(dim=1, keepdim=True) / 7
    q = torch.round(wg / scale).clamp(-7, 7).to(torch.int8)
on_solve: Solved. Step 2 is unlocked. Turn the groups back into one row of floats and compare against one scale per row.
mistakes:
  - { file: quantize_groups/mistakes/placeholders.py, fails: returns_int8, says: "`wg`, `scale` and `q` are still `...`" }
  - { file: quantize_groups/mistakes/wg_placeholder.py, fails: returns_int8, says: "`wg` is still `...`, the placeholder (line 7)" }
  - { file: quantize_groups/mistakes/q_placeholder.py, fails: returns_int8, says: "`q` is still `...`" }
  - { file: quantize_groups/mistakes/float_q.py, fails: returns_int8, says: "Which dtype stores exactly one byte?" }
  - { file: quantize_groups/mistakes/max_tuple.py, fails: returns_int8, says: "returns two things, `(values, indices)`" }
  - { file: quantize_groups/mistakes/no_reshape.py, fails: returns_int8, says: "so `dim=1` doesn't exist" }
  - { file: quantize_groups/mistakes/whole_row.py, fails: one_scale_per_group, says: "one scale for the whole row, like 3.1" }
  - { file: quantize_groups/mistakes/row_scale.py, fails: one_scale_per_group, says: "all 4 groups share it" }
  - { file: quantize_groups/mistakes/no_keepdim.py, fails: one_scale_per_group, says: "lines the 4 scales up with the 4 *columns*" }
  - { file: quantize_groups/mistakes/dim0.py, fails: one_scale_per_group, says: "`dim=0` reduces down each column" }
  - { file: quantize_groups/mistakes/reshape_swapped.py, fails: one_scale_per_group, says: "`reshape(g, -1)` makes g rows" }
  - { file: quantize_groups/mistakes/div127.py, fails: largest_is_7, says: "127, int8's limit from 3.1" }
  - { file: quantize_groups/mistakes/div8.py, fails: largest_is_7, says: "+8 doesn't fit in int4" }
  - { file: quantize_groups/mistakes/div15.py, fails: largest_is_7, says: "*unsigned* 4-bit" }
  - { file: quantize_groups/mistakes/no_abs.py, fails: largest_is_7, says: "Which weight has the largest *magnitude*?" }
  - { file: quantize_groups/mistakes/inverted.py, fails: largest_is_7, says: "Flip it" }
  - { file: quantize_groups/mistakes/multiplied.py, fails: round_trip, says: "You multiplied the weights by `scale`" }
  - { file: quantize_groups/mistakes/truncated.py, fails: round_trip, says: "chops off the decimals" }
```

### Dequantize and compare with per-row

```challenge
id: dequantize_groups
file: groupwise.py
minutes: 25
start: { from: quantize_groups }
setup:
  - say: Your groups quantize. Now go back to floats, and put the 16 weights back into one row so they line up with w.
    type_file: dequantize_groups/stub.py
    pause: 2.0
  - say: "The loop runs g = 16 first. That's one group for the whole row, 3.1's per-row scheme at int4. Then g = 4. Write dequantize_groups and compare the two."
    pause: 1.2
tests: dequantize_groups/tests.py
solution: dequantize_groups/solution.py
alternatives:
  - dequantize_groups/alternatives/flatten.py
  - dequantize_groups/alternatives/int_times_scale.py
hints:
  logic: |
    take q (whole numbers, one row per group)
      and scale (one per row)
      → turn q into floats first   (int8 math overflows)
      → multiply each row by its own scale
      → flatten back into one row, in the original order
      → return it
  tools: |
    turn q into floats
      t.float()          a float32 copy of t
    multiply each row by its scale
      t * x              multiplies every value by x
    flatten into one row
      t.reshape(n)       same values in one row of n; -1 = work it out
  nudge: Two things to undo, the division by scale and the reshape into groups. Which order keeps each group lined up with its own scale?
  concept: "Multiply first, while `q` is `[groups, g]` and `scale` is `[groups, 1]`, so each row meets its own scale. Use `q.float()` to stay out of int8 arithmetic. Then flatten back to one row with `reshape(-1)`."
  example: |
    Continue the trace: `q = [[3, -7, 2, 1], [7, -3, 2, 0]]` and `scale = [[0.1429], [0.5714]]`.

    - As floats times `scale`: row 0 gives `[0.429, -1.0, 0.286, 0.143]`, row 1 gives `[4.0, -1.714, 1.143, 0.0]`. Still `[2, 4]`.
    - Flattened: `[0.429, -1.0, 0.286, 0.143, 4.0, -1.714, 1.143, 0.0]`. That's 8 values, like the original `w`.
    - Compare with `w`: the errors are `[0.029, 0, 0.036, 0.043, 0, 0.186, 0.143, 0.2]`. Row 0's stay under its scale / 2 = 0.071, and row 1's under 0.286.

    Flatten last. If you flatten `q` first, the `[2, 1]` scale can no longer line up with its groups.
  skeleton: |
    return (q.______() * _____).reshape(___)
  solution: |
    return (q.float() * scale).reshape(-1)
on_solve: "Compare the two lines. Mean error drops more than 3× with groups of 4 (0.086 → 0.026). Max error doesn't move: it lives in the 2.40's own group, where three neighbours still share its big scale. Groups contain an outlier's damage, they don't remove it. That's lesson 3.3. Now try g = 2 and g = 8 in the loop: smaller groups, lower error, more scales to store."
mistakes:
  - { file: dequantize_groups/mistakes/placeholder.py, fails: returns_float, says: "still returns `...`" }
  - { file: dequantize_groups/mistakes/no_scale.py, fails: returns_float, says: "is just `q` as floats" }
  - { file: dequantize_groups/mistakes/not_flattened.py, fails: flat_row, says: "one row per group" }
  - { file: dequantize_groups/mistakes/flatten_first.py, fails: flat_row, says: "you flattened `q` before multiplying" }
  - { file: dequantize_groups/mistakes/divided.py, fails: within_bound, says: "You divided by `scale` again" }
  - { file: dequantize_groups/mistakes/global_scale.py, fails: within_bound, says: "ignores the scale it's given" }
  - { file: dequantize_groups/mistakes/no_main.py, fails: script_runs, says: "is missing" }
```

### Quantize your 30M model to int4

```replay
id: quantize_int4_30m
intro: On your GPU this takes a few minutes, most of it the perplexity eval. Watch a recording first so you know what healthy looks like.
log: replays/quantize-int4-g128.jsonl
speed: 40
recorded:
  hardware: RTX 3090
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: Max error per layer
  flag_above: 0.03
  x_labels: [blocks.0, blocks.5]
  caption: blocks.4.mlp.c_proj is still the worst layer. Its outlier now blurs the 127 other weights in its group instead of its whole 2048-wide row. Lesson 3.3 deals with the outlier itself.
summary:
  - { label: Linear weights, value: 37.7 → 9.7 MB }
  - { label: Whole model, value: 60.4 → 32.4 MB }
  - { label: Val perplexity, value: 7.84 → 8.12 }
  - { label: Quality cost, value: +3.6% }
run_yourself: |
  # on your machine or Colab, from the course repo
  python quantize_model.py \
    --ckpt runs/tinystories-30m/best.pt \
    --scheme absmax-int4 --group-size 128 --eval-samples 2000
```

### Record your metrics

```metrics
id: record
intro: One quality metric and one systems metric, from your own run. They're compared with the reference run and the pass limit.
```

## Hints

- Group along each row of an `[out, in]` weight, so `in` must divide by `g`. Every Linear in the 30M model has `in` of 512 or 2048, so 128 fits.
- Common group sizes run from 32 to 128. llama.cpp's Q4_0 uses blocks of 32, and most GPTQ and AWQ checkpoints use 128.
- A group of all zeros gives a scale of 0, and `0 / 0` is NaN. On the GPU, clamp the scale to a tiny minimum, e.g. `scale.clamp(min=1e-8)`.
- PyTorch has no int4 dtype. Store `q` as int8 while you work, and count memory as 4 bits per weight (two packed into each byte) plus one fp16 scale per group. Only the Linear weights are quantized. Embeddings and LayerNorm stay in fp16, as in 3.1.

## Done when

- [ ] Groups of 4 beat one scale per row on the test row {#group_round_trip auto=challenge:dequantize_groups}
- [ ] Every nn.Linear in your 30M model runs in group-wise int4 and still writes readable stories {#model_int4}
- [ ] Perplexity delta and weight memory recorded {#metrics_recorded auto=metrics}
