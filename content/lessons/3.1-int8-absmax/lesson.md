---
id: "3.1"
title: int8 absmax by hand
summary: Write per-row absmax int8 quantization yourself, run it on your 30M model, and measure what it costs.
phase: "3"
hours: 4
type: build
status: ready
protected: true
cut: null
prerequisites: ["2.5", "1.4"]
fallbacks:
  "1.4": If you cut 1.4, quantize the reference TinyStories-30M checkpoint from the course repo instead of your own run.
papers:
  - title: LLM.int8()
    authors: Dettmers et al.
    year: 2022
    url: https://arxiv.org/abs/2208.07339
quality_metric:
  id: ppl_delta
  label: Perplexity change vs fp16
  unit: "%"
  inputs:
    - { id: ppl_fp16, label: fp16 val perplexity, hint: From your baseline eval }
    - { id: ppl_int8, label: int8 val perplexity, hint: "Same eval, quantized" }
  compute: pct_change
  reference: 0.9
  pass: { max: 2.0 }
systems_metric:
  id: weight_mb
  label: int8 weight memory
  unit: MB
  note: The 24 nn.Linear weights plus their row scales. In fp16 they were 37.7 MB. Embeddings stay fp16, so the whole model goes from 60.4 MB to about 41.6 MB.
  inputs:
    - { id: weight_mb, label: int8 weight memory (MB), hint: Linear weights + scales }
  compute: value
  reference: 18.9
  pass: { max: 19.5 }
quiz:
  - id: outlier_row
    q: A row has one weight at 40.0 and every other weight is under 1.0. With per-row absmax, what happens to the small weights?
    answer: 0
    options:
      - text: They collapse onto a handful of levels near zero
        why: scale = 40 / 127 ≈ 0.31, so anything under 0.16 rounds to 0. That's the outlier problem, and it's all of lesson 3.3.
      - text: They're stored exactly; only the 40.0 loses precision
        why: It's the reverse. The outlier sets the scale for the whole row, so the small weights lose the precision.
      - text: int8 overflows and the row turns into NaN
        why: Absmax maps the largest magnitude to exactly 127, so nothing overflows. The damage is silent, which is worse.
  - id: why_127
    q: Why divide by 127 instead of 128?
    answer: 1
    options:
      - text: It's a convention with no effect on accuracy
        why: It matters. With 128, +amax maps to 128, which overflows int8 and wraps to −128.
      - text: So +amax and −amax both land on a value int8 can hold
        why: int8 runs from −128 to 127. Dividing by 127 keeps the grid symmetric around zero and leaves −128 unused.
      - text: 127 is prime, which reduces rounding bias
        why: Primality has nothing to do with it. The limit comes from int8's range of −128…127.
---

## Goal

Write per-row **absmax** int8 quantization by hand: find the biggest magnitude in a row, map it to 127, and scale everything else to match. Then quantize every `nn.Linear` in your TinyStories 30M model and measure what it costs in perplexity and saves in memory.

## Learn

```card
type: concept
id: why_bother
eyebrow: Why bother
title: Llama 3.1 8B needs 16 GB of weights in fp16. Your GPU has 12.
visual:
  kind: bars
  max: 16
  limit: { value: 12, label: 12 GB VRAM }
  bars:
    - { label: fp16, value: 16, text: 16 GB }
    - { label: int8, value: 8, text: 8 GB }
    - { label: int4, value: 4, text: 4 GB }
body: |
  int8 stores each weight in one byte, plus one shared scale per row. This lesson builds the simplest version by hand: **absmax**. Find the biggest magnitude, map it to 127, and scale everything else to match.
```

```card
type: slider_viz
id: fewest_bits
eyebrow: Play with it
title: Find the fewest bits that keep mean error under 0.05.
viz: quant_grid
params:
  weights: [0.42, -1.30, 0.07, 0.88, -0.25, 2.10, -0.61, 0.33, -0.95, 0.15, 1.12, -0.04]
  min_bits: 2
  max_bits: 8
  start_bits: 8
  error_limit: 0.05
goal: fewest_bits_under_limit
answer: 5
body: Top row, the original weights. Bottom row, where each one lands on the int grid. Fewer bits means fewer places to land. Press Check when you think you've found it.
hint: Each bit you remove halves the number of levels, so the gap between them doubles. Step down from 8 and watch where the error turns red.
```

```card
type: fill_blank
id: absmax_scale
eyebrow: Fill the blanks
title: Complete the absmax scale.
code: |
  def quantize(w):
      scale = [[0]] / [[1]]
      q = torch.round(w / scale)
      return q.clamp(-127, 127), scale
tokens: [w.max(), "127", w.abs().max(), "255", w.mean(), "128"]
answer: [w.abs().max(), "127"]
tips:
  w.max(): "`w.max()` ignores big negative weights. Use the magnitude: `w.abs().max()`."
  w.mean(): The mean doesn't bound anything. The scale must come from the largest magnitude.
  "255": int8 holds −128…127. Dividing by 255 would push values past 127.
  "128": +128 doesn't fit in int8. The positive limit is 127.
why_ok: The biggest magnitude maps to exactly ±127, so every weight fits in int8 and nothing clips.
why_wrong: "Swap the order: the magnitude goes on top, the int8 limit below."
hint: The scale has to come from the weight that's biggest in size, whether it's positive or negative. The int8 limit goes underneath.
```

```card
type: predict
id: predict_round
eyebrow: Predict the output
title: Using the quantize() you just wrote, what prints?
code: |
  w = torch.tensor([0.5, -1.0, 0.25])
  q, scale = quantize(w)
  print(q.tolist())
options:
  - text: "[63, -127, 32]"
    why: Close. 0.5 × 127 = 63.5, and torch.round rounds halves to the nearest even number, so it lands on 64.
  - text: "[64, -127, 32]"
    correct: true
    why: amax is 1.0, so scale = 1/127. 0.5 → 63.5 → 64 (round half to even), −1.0 → exactly −127, 0.25 → 31.75 → 32.
  - text: "[0, -1, 0]"
    why: That's rounding w itself. You divide by the scale first, which stretches the row across −127…127.
  - text: "[127, -255, 64]"
    why: −255 can't exist in int8. The largest magnitude always maps to 127.
hint: amax here is 1.0, so scale is 1/127. Multiply each weight by 127, then round. Watch the value that lands exactly on .5.
```

```card
type: reorder
id: round_trip_order
eyebrow: Put it in order
title: Arrange the round trip from weights to error.
lines:
  - amax = w.abs().max()
  - scale = amax / 127
  - q = torch.round(w / scale).to(torch.int8)
  - w_hat = q.float() * scale
  - err = (w - w_hat).abs().mean()
shuffle: [3, 0, 4, 2, 1]
why_ok: "Each line needs the one before it: scale needs amax, q needs scale, w_hat needs q, and the error needs w_hat."
why_wrong: "Something runs before its input exists. Follow the variables: what does each line need?"
hint: Follow the variable names. Each line can only use names that an earlier line created.
```

```card
type: calc
id: llama_int8_gb
eyebrow: Do the math
title: How many GB does Llama 3.1 8B take in int8?
given:
  - { label: Parameters, value: 8.0 × 10⁹ }
  - { label: Bytes per int8 weight, value: "1" }
  - { label: Scales, value: ignore them }
unit: GB
answer: { min: 7.4, max: 8.1 }
placeholder: "0"
why_ok: 8.0 × 10⁹ × 1 byte = 8.0 GB (7.45 GiB). One fp16 scale per 4096-wide row adds about 0.05%.
why_wrong: Each int8 weight is one byte, so the GB figure equals the parameter count in billions.
hint: One int8 weight is one byte, and 10⁹ bytes is 1 GB.
```

```card
type: quiz
ref: outlier_row
```

```card
type: quiz
ref: why_127
```

## Build steps

### Write quantize_absmax

```challenge
id: quantize_absmax
file: absmax.py
minutes: 30
start: { code: "import torch\n\n" }
setup:
  - say: Start with one row of weights. Eight numbers is enough to see everything that happens.
    type: "w = torch.tensor([0.42, -1.30, 0.07, 0.88, -0.25, 2.10, -0.61, 0.33])\n\n"
    pause: 2.6
  - say: "The plan: find the weight with the largest magnitude, map it to 127, and scale the rest of the row by the same amount."
    pause: 3.8
  - say: "Here's the stub. Your turn: fill in scale and q, then run the tests."
    type_file: quantize_absmax/stub.py
    pause: 1.2
tests: quantize_absmax/tests.py
solution: quantize_absmax/solution.py
alternatives:
  - quantize_absmax/alternatives/amax_var.py
  - quantize_absmax/alternatives/function_style.py
  - quantize_absmax/alternatives/no_clamp.py
  - quantize_absmax/alternatives/to_dtype_kwarg.py
hints:
  logic: |
    find the weight with the largest size (ignore the sign)
      → scale = that size ÷ 127   (what one int8 step is worth)
    for each weight in the row:
      divide by scale → round to a whole number
      outside −127…127?
        yes → pull it back to the edge
        no  → keep it
    store the whole numbers in one byte each
      → return q and scale
  tools: |
    find the largest size
      t.abs()          each value without its sign
      t.max()          the biggest value in t
    divide → round to a whole number
      t / x            divides every value by x
      torch.round(t)   nearest whole number
    outside the range? pull it back
      t.clamp(lo, hi)  squeezes values into [lo, hi]
    store in one byte each
      t.to(dtype)      converts; torch.int8 is 1 byte
  nudge: One weight decides the scale for the whole row. Which one, and which int8 value should it land on?
  concept: You want `w / scale` to hit exactly ±127 at the largest |w|. Solve that for scale. Then q is `w / scale` snapped to the nearest whole number, kept inside ±127 and stored in one byte.
  example: |
    Trace it on a tiny row, `w = [0.5, -2.0, 1.0]`:

    - Magnitudes: `[0.5, 2.0, 1.0]`. The largest is `2.0`.
    - `scale` is 2.0 / 127 ≈ `0.01575`.
    - `w / scale` ≈ `[31.75, -127.0, 63.5]`.
    - Rounded: `[32, -127, 64]`.

    Check: the biggest weight lands exactly on −127, and nothing falls outside ±127.
  skeleton: |
    scale = w.____().max() / ___
    q = torch.____(w / scale).clamp(____, ____).to(torch.____)
  solution: |
    scale = w.abs().max() / 127
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
on_solve: Solved. Step 2 is unlocked. Write the way back to floats.
mistakes:
  - { file: quantize_absmax/mistakes/placeholders.py, fails: returns_int8, says: "`scale` and `q` are still `...`" }
  - { file: quantize_absmax/mistakes/scale_placeholder.py, fails: returns_int8, says: "`scale` is still `...`, the placeholder (line 7)" }
  - { file: quantize_absmax/mistakes/q_placeholder.py, fails: returns_int8, says: "`q` is still `...`" }
  - { file: quantize_absmax/mistakes/missing_scale_return.py, fails: returns_int8, says: "return q, scale" }
  - { file: quantize_absmax/mistakes/float_q.py, fails: returns_int8, says: "Which dtype stores exactly one byte?" }
  - { file: quantize_absmax/mistakes/no_abs.py, fails: largest_is_127, says: "your scale comes from 1.0, not −3.0" }
  - { file: quantize_absmax/mistakes/inverted.py, fails: largest_is_127, says: "Flip it" }
  - { file: quantize_absmax/mistakes/mean.py, fails: largest_is_127, says: "The mean doesn't bound anything" }
  - { file: quantize_absmax/mistakes/div128_wraps.py, fails: largest_is_127, says: "+128 doesn't fit in int8" }
  - { file: quantize_absmax/mistakes/multiplied.py, fails: round_trip, says: "You multiplied `w` by `scale`" }
  - { file: quantize_absmax/mistakes/truncated.py, fails: round_trip, says: "chops off the decimals" }
```

### Write dequantize and measure

```challenge
id: dequantize
file: absmax.py
minutes: 20
start: { from: quantize_absmax }
setup:
  - say: Your quantizer passes. The matmul still needs real numbers, so now go back the other way.
    type_file: dequantize/stub.py
    pause: 1.8
  - say: "Your turn: write dequantize. When it's right, max err stays under the bound."
    pause: 1.0
tests: dequantize/tests.py
solution: dequantize/solution.py
alternatives:
  - dequantize/alternatives/int_times_scale.py
hints:
  logic: |
    take q (int8 whole numbers) and its scale
      → turn q into floats first   (int8 math overflows)
      → multiply by scale          (undoes the divide)
      → return the floats
  tools: |
    turn q into floats
      t.float()        a float32 copy of t
    multiply by scale
      t * x            multiplies every value by x
  nudge: Quantizing divided w by scale. What undoes a division?
  concept: Multiply the integers by the same scale. Turn q into floats first so you stay out of int8 arithmetic, where 127 × 2 overflows.
  example: |
    Continue the tiny row from step 1: `q = [32, -127, 64]` and `scale ≈ 0.01575`.

    - As floats: `[32.0, -127.0, 64.0]`.
    - Times `scale`: ≈ `[0.504, -2.0, 1.008]`.
    - Against the original `[0.5, -2.0, 1.0]`, each error is at most half a step (`scale / 2 ≈ 0.0079`).
  skeleton: |
    return q.____() * _____
  solution: |
    return q.float() * scale
on_solve: That's absmax, start to finish. Now experiment. Change both 127s in quantize_absmax to 7 and run again. That's roughly int4. max err grows, and the console notes that step 1's ±127 check no longer holds.
mistakes:
  - { file: dequantize/mistakes/placeholder.py, fails: returns_float, says: "still returns `...`" }
  - { file: dequantize/mistakes/no_scale.py, fails: returns_float, says: "is just `q` as floats" }
  - { file: dequantize/mistakes/divided.py, fails: within_bound, says: "You divided by `scale` again" }
  - { file: dequantize/mistakes/global_scale.py, fails: within_bound, says: "ignores the scale it's given" }
  - { file: dequantize/mistakes/global_w.py, fails: within_bound, says: "ignores the scale it's given" }
  - { file: dequantize/mistakes/no_main.py, fails: script_runs, says: "are missing" }
```

### Quantize your 30M model

```replay
id: quantize_30m
intro: On your GPU this takes a few minutes, most of it the perplexity eval. Watch a recording first so you know what healthy looks like.
log: replays/quantize-30m.jsonl
speed: 40
recorded:
  hardware: RTX 3090
  illustrative: true
  note: Sample data written for the prototype, not a real run. Replace it with a recording of your own run.
chart:
  title: Max error per layer
  flag_above: 0.02
  x_labels: [blocks.0, blocks.5]
  caption: One red bar is an outlier row. A single huge weight sets the scale for everything else in its row. Lesson 3.3 fixes it.
summary:
  - { label: Linear weights, value: 37.7 → 18.9 MB }
  - { label: Whole model, value: 60.4 → 41.6 MB }
  - { label: Val perplexity, value: 7.84 → 7.91 }
  - { label: Quality cost, value: +0.9% }
run_yourself: |
  # on your machine or Colab, from the course repo
  python quantize_model.py \
    --ckpt runs/tinystories-30m/best.pt \
    --scheme absmax-int8 --eval-samples 2000
```

### Record your metrics

```metrics
id: record
intro: One quality metric and one systems metric, from your own run. They're compared with the reference run and the pass limit.
```

## Hints

- If perplexity jumps by more than a few percent, look for one layer whose max error is far above the rest. That's an outlier row, the subject of 3.3.
- Quantize per row (`dim=1` on a `[out, in]` weight), not per tensor. One scale for a whole matrix lets a single outlier ruin every row.
- Count memory as int8 weights plus one fp16 scale per row. Embeddings and LayerNorm stay in fp16 for this lesson.

## Done when

- [ ] Round trip on a test tensor stays within scale / 2 {#round_trip auto=challenge:dequantize}
- [ ] Every nn.Linear in your 30M model runs in int8 and still writes readable stories {#model_int8}
- [ ] Perplexity delta and weight memory recorded {#metrics_recorded auto=metrics}
