---
id: "3.4"
title: GPTQ
summary: Quantize a layer column by column and push each column's rounding error onto the columns still to come, using the inputs' Hessian. Then run GPTQ on Llama 3.1 8B.
phase: "3"
hours: 5
type: run
status: ready
protected: true
cut: null
prerequisites:
  - "3.3"
papers:
  - title: GPTQ
    authors: Frantar et al.
    year: 2022
    url: https://arxiv.org/abs/2210.17323
quality_metric:
  id: ppl_delta
  label: GPTQ int4 perplexity change vs fp16
  unit: "%"
  note: int4, groups of 128, 128 calibration samples. Round-to-nearest with the same groups was +8.8% in the reference run.
  inputs:
    - { id: ppl_fp16, label: fp16 val perplexity, hint: "Llama 3.1 8B, with CPU offload if it doesn't fit" }
    - { id: ppl_gptq, label: GPTQ int4 g128 perplexity, hint: "Same eval, quantized" }
  compute: pct_change
  reference: 4.5
  pass: { max: 6.0 }
systems_metric:
  id: quant_minutes
  label: Time to quantize
  unit: min
  note: Wall-clock for all 32 blocks, calibration forward passes included.
  inputs:
    - { id: quant_minutes, label: Minutes to quantize, hint: From the run's last line }
  compute: value
  reference: 21
  pass: { max: 60 }
quiz:
  - id: weight_vs_output
    q: On the lesson's layer, GPTQ's mean weight error is higher than round-to-nearest's (0.0158 vs 0.0127), yet its output error is 7× lower. How?
    answer: 1
    options:
      - text: GPTQ rounds more accurately than round-to-nearest
        why: Nothing rounds more accurately than rounding to the nearest level. Each weight on its own is as close as it can get under RTN.
      - text: It lets weights drift in directions the inputs can't see
        why: The loss is X @ W.T, not W. When features move together, an error on one can be cancelled by nudging a correlated one, so GPTQ trades a little weight error for a lot less output error.
      - text: It quantizes to more levels than int4
        why: Both use the same 15 levels and the same per-row scale. Only the choice of level for each weight differs.
  - id: why_damp
    q: Why does GPTQ add 1% of the diagonal's mean to H's diagonal before inverting it?
    answer: 0
    options:
      - text: A feature that never fires makes H singular, and small diagonals make the updates explode
        why: A dead feature leaves a zero row and column in H, so H has no inverse. Near-zero diagonals are almost as bad, giving huge U entries. A small damping term keeps the inverse defined and the updates sane.
      - text: It speeds up the Cholesky decomposition
        why: Cholesky costs the same either way. Without damping it can fail outright, which is the real problem.
      - text: It makes GPTQ reproducible across runs
        why: GPTQ is deterministic for fixed calibration data. Damping is about H being invertible and well-conditioned.
  - id: calibration_text
    q: You're going to GPTQ-quantize a model fine-tuned on Mongolian. What should the 128 calibration samples be?
    answer: 2
    options:
      - text: Random token ids, so no language is favoured
        why: Random tokens produce activations unlike anything the model sees in use, so H describes the wrong input directions and the compensation goes to waste.
      - text: English C4, the standard GPTQ calibration set
        why: C4 is the default for English models. For a Mongolian model, English text exercises different features, and the layer gets tuned for inputs it won't see.
      - text: Mongolian text like the model will see in use
        why: H = 2 XᵀX / n is built from the calibration inputs, so GPTQ protects the output on inputs like those. Calibrate on the distribution you'll serve. That's lesson 4A's corpus.
---

## Goal

Round-to-nearest (RTN) quantizes every weight on its own. **GPTQ** quantizes one input column at a time and, after each, nudges the columns not yet quantized to cancel the error it just made, as seen through real calibration inputs. The information it needs is the Hessian of the layer's output error, H = 2XᵀX / n. Build H with GPTQ's damping, write the column loop, and compare output error against RTN on a tiny layer. Then run GPTQ on Llama 3.1 8B, which doesn't fit in 12 GB at fp16 but quantizes fine one block at a time.

## Learn

```card
type: concept
id: output_not_weights
eyebrow: The idea
title: What matters is X @ W.T, not W.
visual:
  kind: bars
  max: 0.01
  bars:
    - { label: RTN, value: 0.0092, text: "0.0092" }
    - { label: GPTQ, value: 0.0013, text: "0.0013" }
body: |
  When features move together, rounding one weight up can be cancelled by rounding a correlated one down, and the layer's output barely changes. GPTQ quantizes the input columns in order and, after each one, adds the error it just made onto the columns still to come, weighted by how the calibration inputs correlate. The bars show output error on this lesson's 4 × 8 layer at int4.
```

```card
type: predict
id: predict_compensate
eyebrow: Predict the output
title: Features 0 and 1 are always equal, so the layer only sees w0 + w1. What does GPTQ produce?
code: |
  # grid step 0.25, so a weight can be 0.25 or 0.50, not 0.36
  w = [0.36, 0.36]          # w0 + w1 = 0.72
  # GPTQ: round w0, then add its error to w1, then round w1
options:
  - text: "[0.25, 0.25]"
    why: That's round-to-nearest, both weights alone. The sum is 0.50 instead of 0.72, an error of 0.22 on every token.
  - text: "[0.25, 0.50]"
    correct: true
    why: "w0 rounds to 0.25, an error of 0.11. Since x1 = x0, adding that 0.11 to w1 keeps the output the same: 0.36 + 0.11 = 0.47, which rounds to 0.50. The sum is 0.75, off by just 0.03."
  - text: "[0.50, 0.50]"
    why: 0.36 is closer to 0.25 than to 0.50, so w0 rounds down. Only w1, after picking up w0's error, rounds up.
  - text: "[0.36, 0.36]"
    why: Those aren't on the grid. Every weight has to land on a level, and GPTQ only chooses which one.
hint: Round w0 first and note how far it moved. Because x1 always equals x0, moving w1 by the same amount in the other direction keeps the output unchanged.
```

```card
type: fill_blank
id: gptq_update
eyebrow: Fill the blanks
title: Complete one step of GPTQ's column loop.
code: |
  q = torch.round(W[:, j] / scale).clamp(-7, 7) * scale
  err = (W[:, j] - q) / [[0]]
  W[:, j:] -= err[:, None] * [[1]]
tokens: ["U[j, j]", "U[j, j:]", "H[j, j]", "U[:, j]", "H[j, j:]"]
answer: ["U[j, j]", "U[j, j:]"]
tips:
  "H[j, j]": H itself says how strongly features correlate, not how to correct for them. The update comes from its inverse, through U, the Cholesky factor of H⁻¹.
  "U[:, j]": That's a column of U, and U is upper-triangular, so it only reaches the columns *before* j. The correction goes to the columns after j, which is row j.
  "H[j, j:]": The correction weights come from H⁻¹ (through U), not H. H tells you how features correlate; its inverse tells you how to cancel an error.
why_ok: "U is the upper Cholesky factor of H⁻¹. Row j of U, from column j on, says how much of column j's error each later column should absorb, and dividing by U[j, j] puts the error on that scale."
why_wrong: The update comes from row j of U, the Cholesky factor of H's inverse. One entry scales the error, and the rest of the row spreads it over the later columns.
hint: Everything GPTQ needs comes from U, and column j's error only moves onto columns j and later. Which entries of U sit in row j?
```

```card
type: reorder
id: gptq_pipeline
eyebrow: Put it in order
title: GPTQ on a real model, one transformer block at a time.
lines:
  - Run the calibration samples through the blocks quantized so far
  - Record the inputs X of each Linear in this block
  - Build H = 2XᵀX / n for each Linear and add damping
  - Quantize each Linear column by column, compensating as you go
  - Swap in the quantized weights and move to the next block
shuffle: [2, 4, 0, 3, 1]
why_ok: "Inputs, Hessian, quantize, replace, repeat. Each block sees inputs that already passed through the quantized blocks before it, so it corrects for their errors too, and only one block's weights and Hessians are in memory at once."
why_wrong: GPTQ can't build H without X, and X has to come from the blocks before this one, already quantized. Follow what each step needs.
hint: H is built from X, and the column loop needs H. Where does X come from?
```

```card
type: calc
id: hessian_mb
eyebrow: Do the math
title: How many MB is H for Llama 3.1 8B's down_proj, stored in fp32?
given:
  - { label: down_proj inputs, value: "14,336" }
  - { label: H shape, value: "[inputs, inputs]" }
  - { label: Bytes per fp32 value, value: "4" }
unit: MB
answer: { min: 780, max: 825 }
placeholder: "0"
why_ok: 14,336² × 4 bytes = 822 MB (784 MiB), for one Linear. A block has seven of them, so GPTQ builds Hessians one block at a time and frees them before moving on.
why_wrong: H has one row and one column per input feature, so it holds 14,336 × 14,336 numbers at 4 bytes each.
hint: Square the number of inputs, multiply by 4 bytes, and divide by 10⁶ for MB.
```

```card
type: quiz
ref: weight_vs_output
```

```card
type: quiz
ref: why_damp
```

```card
type: quiz
ref: calibration_text
```

## Build steps

### Build the Hessian

```challenge
id: hessian
file: gptq.py
minutes: 25
start: { code: "import torch\n\n" }
setup:
  - say: "GPTQ needs real inputs to know which errors matter. Here are 64 calibration tokens. Their 8 features are mixtures of 3 hidden causes plus a little noise, so they move together, the way real activations do."
    type: "g = torch.Generator().manual_seed(6)\nmix = torch.randn(3, 8, generator=g)\nX = torch.randn(64, 3, generator=g) @ mix + 0.1 * torch.randn(64, 8, generator=g)\n"
    pause: 3.4
  - say: Feature 7 never fires in this batch. That happens in real models too, and it will matter in a moment.
    type: "X[:, 7] = 0.0\n"
    pause: 2.0
  - say: And the layer to quantize, four outputs over the eight features.
    type: "W = 0.3 * torch.randn(4, 8, generator=g)\n"
    pause: 1.8
  - say: "The Hessian of the output error is H = 2XᵀX / n: one row and one column per feature, saying how each pair of features moves together. GPTQ adds a little to its diagonal. Your turn: fill in H, damp and the return."
    type_file: hessian/stub.py
    pause: 1.2
tests: hessian/tests.py
solution: hessian/solution.py
alternatives:
  - hessian/alternatives/in_place.py
  - hessian/alternatives/eye_sum.py
  - hessian/alternatives/no_constant.py
hints:
  start:
    try: |
      print(X.shape)
      print((X[:, 0] * X[:, 1]).sum())
      print((X[:, 0] * X[:, 0]).sum())
    see: |
      torch.Size([64, 8])
      tensor(368.3354)
      tensor(243.7676)
    next: "That's the shape of X (64 tokens, 8 features) and two entries of XᵀX: features 0 and 1 together, and feature 0 with itself. `H` holds this number for every pair of features, so it is 8 by 8. One matrix product of X with itself gives the whole grid, with the 64 tokens as the part that gets added up. Then scale by 2 over the token count."
  logic: |
    for every pair of features:
      multiply them token by token, add up over all tokens
    scale by 2 ÷ number of tokens
      → H   (one row and column per feature)
    damp = percdamp × average of H's diagonal
    add damp to each diagonal entry only
      → return H
  tools: |
    products of every feature pair
      t.T              rows and columns swapped
      a @ b            matrix product
    scale by 2 over n
      t.shape[i]       the size of dimension i of t
      plain arithmetic: * /
    damp from the diagonal
      H[idx, idx]      the diagonal entries picked by index
      t.mean()         the average of all values
    add to the diagonal only
      torch.eye(n)     n × n, ones on the diagonal, zeros elsewhere
      plain arithmetic: * +
  nudge: "Two features, i and j, are related through the tokens they share. Which product of X with itself gives one number per *pair of features*?"
  concept: "X is [tokens, features], so `X.T @ X` is [features, features]: entry [i, j] sums feature i times feature j over the tokens. Multiply by 2 and divide by the number of tokens. `H[idx, idx]` picks out the diagonal, so `percdamp * H[idx, idx].mean()` is GPTQ's damping, and adding it times `torch.eye` touches the diagonal only."
  example: |
    Trace it on 2 tokens and 3 features, where feature 2 never fires: `X = [[1, 2, 0], [3, -1, 0]]`.

    - `XᵀX` is [3, 3]. Entry [0, 0] is 1² + 3² = 10, entry [0, 1] is 1×2 + 3×(−1) = −1, entry [1, 1] is 2² + (−1)² = 5. Everything in row and column 2 is 0.
    - With 2 tokens, `2 / n` is 1, so `H = [[10, -1, 0], [-1, 5, 0], [0, 0, 0]]`.
    - The diagonal is `[10, 5, 0]`, its mean is 5, and `damp = 0.01 × 5 = 0.05`.
    - Result: `[[10.05, -1, 0], [-1, 5.05, 0], [0, 0, 0.05]]`. The dead feature's diagonal is no longer 0, so `H` can be inverted.
  skeleton: |
    H = 2 * X.__ @ X / X.shape[___]
    damp = ________ * H[idx, idx].____()
    return H + damp * torch.___(len(idx))
  solution: |
    H = 2 * X.T @ X / X.shape[0]
    damp = percdamp * H[idx, idx].mean()
    return H + damp * torch.eye(len(idx))
on_solve: "Look at row 7 of H. It's all zero except the diagonal, where your damping sits. Without it, H would have no inverse and the next step would divide by zero. The 2 / n in front doesn't change what GPTQ does, only H's overall size, since the update divides one entry of the inverse by another."
mistakes:
  - { file: hessian/mistakes/placeholders.py, fails: feature_square, says: "still returns `...`" }
  - { file: hessian/mistakes/h_placeholder.py, fails: feature_square, says: "`H` is still `...`" }
  - { file: hessian/mistakes/x_xt.py, fails: feature_square, says: "one row per calibration *token*" }
  - { file: hessian/mistakes/no_damp.py, fails: damped, says: "never fires" }
  - { file: hessian/mistakes/every_entry.py, fails: damped, says: "every entry" }
  - { file: hessian/mistakes/scaled_diag.py, fails: damped, says: "stays 0" }
  - { file: hessian/mistakes/absolute_damp.py, fails: damped, says: "whatever H's size" }
  - { file: hessian/mistakes/mean_all.py, fails: damped, says: "off-diagonal entries included" }
  - { file: hessian/mistakes/fixed_percdamp.py, fails: damped, says: "Use the `percdamp` argument" }
```

### Quantize column by column

```challenge
id: gptq
file: gptq.py
minutes: 45
start: { from: hessian }
setup:
  - say: "One int4 scale per row, the same as 3.2 with a whole row per group, fixed before GPTQ starts."
    type: "\nscale = W.abs().amax(dim=1) / 7   # one int4 scale per row\n"
    pause: 2.0
  - say: "GPTQ needs the upper Cholesky factor of H's inverse. On your GPU that's two torch.linalg calls. Mini-torch has no torch.linalg, so here are both, written out."
    type: "\ndef inverse(A):\n    # Gauss-Jordan elimination. On your GPU: torch.linalg.inv(A)\n    n = A.shape[0]\n    M = torch.cat([A.clone(), torch.eye(n)], dim=1)\n    for i in range(n):\n        M[i] = M[i] / M[i, i]\n        for r in range(n):\n            if r != i:\n                M[r] = M[r] - M[r, i] * M[i]\n    return M[:, n:]\n\ndef upper_cholesky(A):\n    # U with A = U.T @ U. On your GPU: torch.linalg.cholesky(A, upper=True)\n    n = A.shape[0]\n    U = torch.zeros_like(A)\n    for i in range(n):\n        U[i, i] = (A[i, i] - (U[:i, i] ** 2).sum()).sqrt()\n        for k in range(i + 1, n):\n            U[i, k] = (A[i, k] - (U[:i, i] * U[:i, k]).sum()) / U[i, i]\n    return U\n"
    pause: 2.6
  - say: "Now the loop. For each column: round it onto each row's grid, work out the error, and push that error onto the columns not yet done. Your turn: fill in q, err and update."
    type_file: gptq/stub.py
    pause: 1.2
tests: gptq/tests.py
solution: gptq/solution.py
alternatives:
  - gptq/alternatives/outer.py
  - gptq/alternatives/next_columns.py
  - gptq/alternatives/unsqueeze.py
hints:
  start:
    try: |
      U = upper_cholesky(inverse(H))
      print(W[:, 0] / scale)
      print(U[0, 0])
    see: |
      tensor([3.7592, 0.0506, 2.9851, 0.2696])
      tensor(2.8465)
    next: "That's column 0 of W measured in steps of each row's scale, and the first diagonal entry of `U`. Round those four numbers, clamp them to ±7 and multiply back, and you have `q` for column 0. `err` is how far column 0 moved, divided by that `U[0, 0]`. Do column 0 by hand before you think about the loop."
  logic: |
    for each column j, left to right:
      divide column j by each row's scale → round to a whole number
        outside −7…7?
          yes → pull it back to the edge
          no  → keep it
      multiply back by scale → q, store it in Q
      err = how far column j moved ÷ U's diagonal entry for j
      update = err (per row) times U's row for j, from j on
      subtract update from columns j onward
        (column j lands on q, later columns adjust)
    return Q
  tools: |
    round onto the grid
      torch.round(t)   nearest whole number
      t.clamp(lo, hi)  squeezes values into [lo, hi]
      plain arithmetic: * /   (to and from the grid)
    how far it moved
      plain arithmetic: - /
      indexing into U: single entries and slices of a row
    the correction
      t[:, None]       adds a column axis, [n] becomes [n, 1]
      a * b            multiplies, a column against a row gives a grid
  nudge: "Three lines per column. Quantize column j exactly like 3.2 did. Then work out how far it moved, and hand that error to the columns after it. Where in U does column j's correction live?"
  concept: "`q` is column j rounded on each row's grid: divide by `scale`, round, clamp to ±7, multiply back. The error is `W[:, j] - q`, one value per row, divided by `U[j, j]`. Row j of U from column j on, `U[j, j:]`, says how much of that error each remaining column absorbs. The update is the outer product of the two, a [rows, columns left] block: `err[:, None] * U[j, j:]`."
  example: |
    Trace it on one row, two features that move together, with a step of `scale = 0.1`: `W = [[0.34, 0.24]]`, `H = [[1, 0.8], [0.8, 1]]`.

    - `U` (upper Cholesky factor of H⁻¹) ≈ `[[1.667, -1.333], [0, 1]]`.
    - j = 0: 0.34 / 0.1 = 3.4, rounds to 3, so `q = 0.3`. The error is 0.04, `err = 0.04 / 1.667 = 0.024`.
    - `update = 0.024 × [1.667, -1.333] = [0.04, -0.032]`. Column 0 becomes 0.34 − 0.04 = 0.3, and column 1 becomes 0.24 + 0.032 = 0.272.
    - j = 1: 0.272 / 0.1 = 2.72, rounds to 3, so `q = 0.3`. Result: `Q = [[0.3, 0.3]]`.
    - Plain rounding would give `[[0.3, 0.2]]`. Against the original `[0.34, 0.24]`, the output error (the weight error `d` measured through `H`, as `d·H·d`) is 0.00136 for GPTQ and 0.00576 for rounding, over 4× lower, even though the second weight moved further (0.06 against 0.04).
  skeleton: |
    q = torch.round(W[:, j] / _____).clamp(___, ___) * _____
    err = (W[:, j] - _) / U[_, _]
    update = err[:, ____] * U[_, __]
  solution: |
    q = torch.round(W[:, j] / scale).clamp(-7, 7) * scale
    err = (W[:, j] - q) / U[j, j]
    update = err[:, None] * U[j, j:]
on_solve: "GPTQ's output error is 7× lower than round-to-nearest's, and its weight error is *higher*. It moved weights in directions the calibration inputs can't see. That's the whole algorithm. The real thing adds one speedup and two accuracy tricks. The speedup is lazy batches: it updates 128 columns at a time before touching the rest of the matrix. For accuracy it uses group scales like 3.2, and can quantize the columns with the largest diagonal of H first (act-order). Both of those cost a little speed."
mistakes:
  - { file: gptq/mistakes/q_placeholder.py, fails: on_grid, says: "`q` is still `...`" }
  - { file: gptq/mistakes/err_placeholder.py, fails: on_grid, says: "`err` is still `...`" }
  - { file: gptq/mistakes/update_placeholder.py, fails: on_grid, says: "`update` is still `...`" }
  - { file: gptq/mistakes/ints.py, fails: on_grid, says: "the integers themselves" }
  - { file: gptq/mistakes/no_round.py, fails: on_grid, says: "isn't a whole number of steps" }
  - { file: gptq/mistakes/no_outer.py, fails: on_grid, says: "`err[:, None]`" }
  - { file: gptq/mistakes/rtn.py, fails: compensates, says: "plain round-to-nearest" }
  - { file: gptq/mistakes/no_divide.py, fails: compensates, says: "never divided by `U[j, j]`" }
  - { file: gptq/mistakes/plus.py, fails: compensates, says: "pile up instead of cancelling" }
  - { file: gptq/mistakes/uses_h.py, fails: compensates, says: "rows of H" }
  - { file: gptq/mistakes/hinv_rows.py, fails: compensates, says: "go stale" }
```

### Run GPTQ on Llama 3.1 8B

```replay
id: gptq_8b
intro: "Llama 3.1 8B is 16 GB in fp16, more than your card holds, but GPTQ only needs one block on the GPU at a time. The run takes about 20 minutes plus an hour of evals. Watch a recording first."
log: replays/gptq-llama-8b.jsonl
speed: 40
recorded:
  hardware: RTX 3060 12 GB
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: GPTQ output error per block (sum over its 7 Linears)
  flag_above: 20
  x_labels: [blocks.0, blocks.31]
  caption: "Error grows with depth, because later blocks see larger activations and inherit every earlier block's error. The last two blocks are the hardest, as in most Llama GPTQ logs. This is squared output error, so compare blocks with each other, not with 3.2's max error."
summary:
  - { label: Time to quantize, value: 21 min }
  - { label: RTN int4 g128, value: "6.24 → 6.79 (+8.8%)" }
  - { label: GPTQ int4 g128, value: "6.24 → 6.52 (+4.5%)" }
  - { label: Size on disk, value: "16.1 → 5.7 GB" }
run_yourself: |
  # on your machine or Colab, from the course repo
  python gptq_quantize.py --model meta-llama/Llama-3.1-8B \
    --bits 4 --group-size 128 --calib c4 --nsamples 128 --damp 0.01
  python eval_ppl.py --model out/llama-3.1-8b-gptq-g128 --eval-samples 2000
```

### Record your metrics

```metrics
id: record
intro: GPTQ's perplexity change against fp16, and how long the quantization took on your card.
```

## Hints

- GPTQ never needs the whole fp16 model on the GPU. Load blocks one at a time (the GPTQ libraries do this for you). Only the fp16 *baseline* eval needs the full model: run it once with `device_map="auto"`, which offloads layers to CPU memory. It's slow but it fits.
- If quantization dies with "Cholesky decomposition failed" or the output turns to NaN, raise the damping from 0.01 to 0.1. Some layers have nearly dead features, exactly like feature 7 here.
- Use 128 samples of 2048 tokens from text like your eval set. More samples help a little and cost time linearly.
- Act-order (`desc_act=True`) quantizes the columns with the largest diagonal of H first, while there are the most columns left to absorb the error. It usually helps a few tenths of a percent.

## Done when

- [ ] GPTQ beats round-to-nearest on the lesson's layer {#gptq_beats_rtn auto=challenge:gptq}
- [ ] Llama 3.1 8B quantized with GPTQ int4 g128 and writing coherent text {#model_gptq}
- [ ] Perplexity change and quantization time recorded {#metrics_recorded auto=metrics}
