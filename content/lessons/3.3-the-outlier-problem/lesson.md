---
id: "3.3"
title: The outlier problem
summary: Find the activation outliers that break int8 matmuls, then fix them two ways, LLM.int8()'s mixed-precision split and SmoothQuant's scale migration.
phase: "3"
hours: 4
type: research
status: ready
protected: true
cut: null
prerequisites:
  - "3.2"
papers:
  - title: LLM.int8()
    authors: Dettmers et al.
    year: 2022
    url: https://arxiv.org/abs/2208.07339
  - title: SmoothQuant
    authors: Xiao et al.
    year: 2022
    url: https://arxiv.org/abs/2211.10438
quality_metric:
  id: ppl_delta
  label: Perplexity change after outlier handling
  unit: "%"
  note: W8A8 (int8 weights and int8 activations) with SmoothQuant, against the same model in fp16.
  inputs:
    - { id: ppl_fp16, label: fp16 val perplexity, hint: Llama 3.2 1B on your eval set }
    - { id: ppl_w8a8, label: SmoothQuant W8A8 perplexity, hint: "Same eval, alpha 0.5" }
  compute: pct_change
  reference: 0.6
  pass: { max: 2.0 }
systems_metric:
  id: outlier_cols
  label: Outlier feature dimensions found
  unit: count
  note: Hidden dimensions where |x| > 6 in at least a quarter of the layers, LLM.int8()'s criterion.
  inputs:
    - { id: outlier_cols, label: Outlier dimensions, hint: From the scan's summary line }
  compute: value
  reference: 6
  pass: { min: 1 }
quiz:
  - id: why_not_clip
    q: Outlier features are only a handful of the 2048 hidden dimensions. Why not clip them to ±6 and quantize everything to int8?
    answer: 1
    options:
      - text: Clipping is slower than splitting the matmul
        why: Clipping is one cheap elementwise op. The problem is what it does to the model, not its speed.
      - text: The model relies on them, and clipping wrecks its output
        why: Dettmers et al. zeroed the outlier dimensions and perplexity exploded, even though they're a tiny fraction of the features. They carry signal the model depends on, so they have to survive quantization.
      - text: int8 can't represent 6
        why: int8 can represent any range you like, because the scale stretches it. The problem is that a large range makes the steps coarse for everything else.
  - id: why_weights_ok
    q: SmoothQuant divides the activations by s and multiplies the weights by s. Why doesn't that just move the outlier into the weights?
    answer: 0
    options:
      - text: It moves part of it, and weights have plenty of room to spare
        why: Weights have no outliers to start with, so they can absorb a moderate factor. With alpha = 0.5 both sides end up with the same max for that feature, √(max|X| · max|W|), and neither one is extreme.
      - text: It does move all of it, but weights are kept in fp16
        why: SmoothQuant quantizes the weights to int8 too. That's the whole point, W8A8. It only works because the migrated scale stays moderate on both sides.
      - text: The factor cancels out before quantization
        why: It cancels in the exact product X @ W.T, which is why the trick is safe. But each side is quantized on its own, with the factor applied, and that's where it helps.
  - id: fp16_share
    q: A 2048-wide layer has 6 outlier dimensions. With LLM.int8()'s split, what share of the matmul runs in fp16?
    answer: 2
    options:
      - text: About half, the outlier tokens
        why: The split is by feature column, not by token. Every token goes through both parts, but the fp16 part only covers the 6 outlier columns.
      - text: All of it, and int8 only stores the weights
        why: That's weight-only quantization. LLM.int8() really multiplies int8 by int8 for the regular columns.
      - text: About 0.3%, 6 of 2048 columns
        why: 6 / 2048 ≈ 0.3% of the inner dimension. The fp16 part is cheap in arithmetic, but splitting, gathering and adding two results costs time, so LLM.int8() saves memory without being faster than fp16.
---

## Goal

In 3.1 and 3.2 you quantized weights. A matmul has a second operand, the activations, and in large language models a handful of activation **features** carry values 20 to 100 times bigger than the rest, in nearly every token. Quantize activations to int8 per token and each token's scale is set by its outlier, so all its other features land on a few levels. Write LLM.int8()'s vector-wise int8 matmul, watch the outlier wreck it, then fix it twice: split the outlier columns off into full precision (**LLM.int8()**), and migrate the outlier's scale into the weights (**SmoothQuant**). Finally scan a real model for outlier features.

## Learn

```card
type: concept
id: one_feature
eyebrow: Why it breaks
title: One feature, huge in every token, sets every token's scale.
visual:
  kind: bars
  max: 0.025
  bars:
    - { label: int8, value: 0.0239, text: "0.024" }
    - { label: SmoothQuant, value: 0.0027, text: "0.0027" }
    - { label: LLM.int8(), value: 0.0012, text: "0.0012" }
body: |
  Dettmers et al. found that from about 6.7B parameters on, a few hidden dimensions hold values around 20 to 60 in almost every token, while the rest stay near 1. Quantize a token per row with absmax and its scale comes from the outlier, so a feature at 0.4 lands on 3 out of 127. The bars show mean output error of a 4-token, 6-feature int8 matmul from this lesson, and of the two fixes you'll write.
```

```card
type: slider_viz
id: token_bits
eyebrow: Play with it
title: One token's 12 features, one of them at −18.5. Fewest bits for mean error under 0.02?
viz: quant_grid
params:
  weights: [0.42, -0.61, -18.5, 0.27, 0.95, 0.33, -0.88, 0.14, 0.52, -0.31, 0.71, -0.45]
  min_bits: 2
  max_bits: 12
  start_bits: 8
  error_limit: 0.02
goal: fewest_bits_under_limit
answer: 9
body: These are activations this time, not weights. It starts at 8 bits, int8. Without the −18.5, the other 11 features need only 5 bits for the same error.
hint: At 8 bits the step is 18.5 / 127 ≈ 0.15, so features under 0.07 round to 0. Go up one bit at a time and watch the small ones find their own levels.
```

```card
type: predict
id: predict_token
eyebrow: Predict the output
title: 3.1's per-row absmax, on one token with an outlier. What prints?
code: |
  x = torch.tensor([[0.42, -0.61, -18.5, 0.27]])
  scale = x.abs().amax(dim=1, keepdim=True) / 127
  q = torch.round(x / scale).clamp(-127, 127)
  print(q.tolist())
options:
  - text: "[[127.0, -127.0, -127.0, 127.0]]"
    why: That would need a separate scale per feature. Here one scale covers the whole row, and it comes from −18.5.
  - text: "[[3.0, -4.0, -127.0, 2.0]]"
    correct: true
    why: "scale = 18.5 / 127 ≈ 0.146. 0.42 → 2.88 → 3, −0.61 → −4.19 → −4, 0.27 → 1.85 → 2. Three of the four features share 9 levels between them."
  - text: "[[0.0, 0.0, -127.0, 0.0]]"
    why: Not quite that bad at int8. 0.42 / 0.146 ≈ 2.9, which rounds to 3. At int4, with a step of 18.5 / 7 ≈ 2.6, they would all be 0.
  - text: "[[42.0, -61.0, -127.0, 27.0]]"
    why: That's what you'd get with a scale of 0.01. The scale is set by the largest magnitude, 18.5.
hint: Work out the scale first. Which value has the largest magnitude, and what does it map to?
```

```card
type: fill_blank
id: mixed_split
eyebrow: Fill the blanks
title: Complete LLM.int8()'s mixed-precision split.
code: |
  def mixed_matmul(X, W, threshold=6.0):
      big = (X.abs() > threshold).[[0]](dim=[[1]])
      return X[:, big] @ W[:, big].T + int8_matmul(X[:, [[2]]], W[:, ~big])
tokens: [all, any, "0", "1", big, ~big]
answer: [any, "0", ~big]
tips:
  all: "`all` only flags a feature when *every* token is over the threshold. One token with a 20 is enough to wreck that token's scale."
  "1": "`dim=1` reduces across the features, giving one flag per token. The split is by feature column, so reduce over the tokens."
  big: That sends the outlier columns through int8 a second time. The int8 path gets everything *except* the outliers.
why_ok: "`big` has one flag per feature. The outlier columns multiply in full precision, every other column in int8, and the two partial products add up to the whole matmul."
why_wrong: "The mask needs one flag per feature column, True when any token is over the threshold. Then the int8 path takes the columns that aren't flagged."
hint: "X is [tokens, features]. You want one flag per feature, so reduce over the tokens. Which reduction is True if *at least one* value is?"
```

```card
type: calc
id: smooth_factor
eyebrow: Do the math
title: SmoothQuant with alpha = 0.5. What's this feature's factor s?
given:
  - { label: "max |X_j| over tokens", value: "16" }
  - { label: "max |W_j| over outputs", value: "0.25" }
  - { label: Factor, value: "s = max|X_j|^α / max|W_j|^(1−α)" }
unit: "×"
answer: { min: 7.99, max: 8.01 }
placeholder: "0"
why_ok: √16 / √0.25 = 4 / 0.5 = 8. Afterwards the activations' max is 16 / 8 = 2 and the weights' max is 0.25 × 8 = 2. Alpha = 0.5 splits the difficulty evenly.
why_wrong: With alpha = 0.5 both powers are square roots, so s = √16 / √0.25.
hint: Raise each max to its power first. With alpha = 0.5, both powers are 0.5, a square root.
```

```card
type: quiz
ref: why_not_clip
```

```card
type: quiz
ref: why_weights_ok
```

```card
type: quiz
ref: fp16_share
```

## Build steps

### Write an int8 × int8 matmul

```challenge
id: int8_matmul
file: outliers.py
minutes: 30
start: { code: "import torch\n\n" }
setup:
  - say: In 3.1 and 3.2 you quantized weights. A matmul has a second operand, and that's where the trouble lives. Here are four tokens with six features each. Look at feature 2.
    type: "X = torch.tensor([\n    [0.42, -0.61, -18.5, 0.27, 0.95, 0.33],\n    [-0.88, 0.14, -21.0, -0.52, 0.31, 9.6],\n    [0.35, 0.72, -19.2, 0.60, -0.44, 0.21],\n    [-0.27, -0.95, -17.8, 0.18, 0.66, 8.7],\n])\n"
    pause: 3.0
  - say: And a small layer, three outputs over those six features. Rows are output channels, like nn.Linear's weight, so the layer computes X @ W.T.
    type: "W = torch.tensor([\n    [0.21, -0.43, 0.05, 0.37, -0.12, 0.08],\n    [-0.30, 0.18, -0.07, 0.25, 0.41, -0.06],\n    [0.15, 0.33, 0.04, -0.28, 0.19, 0.11],\n])\n\n"
    pause: 2.4
  - say: 3.1's quantizer, unchanged. One absmax scale per row.
    type: "def quantize_rows(t):\n    # 3.1's absmax int8, one scale per row\n    scale = t.abs().amax(dim=1, keepdim=True) / 127\n    q = torch.round(t / scale).clamp(-127, 127).to(torch.int8)\n    return q, scale\n"
    pause: 2.0
  - say: "LLM.int8()'s vector-wise matmul quantizes X per token and W per output channel, multiplies the integers, and scales the result back. Your turn: fill in acc and the return."
    type_file: int8_matmul/stub.py
    pause: 1.2
tests: int8_matmul/tests.py
solution: int8_matmul/solution.py
alternatives:
  - int8_matmul/alternatives/long_acc.py
  - int8_matmul/alternatives/float_acc.py
  - int8_matmul/alternatives/outer_scales.py
hints:
  nudge: "Two things to get right: the integer products must not overflow, and output [i, j] came from token i and channel j. Which two scales does it need?"
  concept: "int8 × int8 products summed over six features run into the thousands, past int8's 127, so real kernels accumulate in int32: cast with `.int()` before `@`. `acc` is [4, 3]. `sx` is a [4, 1] column, one scale per token, which lines up with acc's rows. `sw` is [3, 1], one per output channel, and output channels run along acc's columns, so it needs to be a row: `sw.T`."
  skeleton: |
    acc = qx.____() @ qw.____().___
    return acc.float() * ___ * ___.___
  solution: |
    acc = qx.int() @ qw.int().T
    return acc.float() * sx * sw.T
on_solve: "Look at the int8 X that printed. Column 2 is −127 in every token, and the other features land on single digits, so each token keeps only a few levels for everything that isn't the outlier. Mean output error is 0.024 on outputs around 1. Next: get the outlier out of the int8 path."
mistakes:
  - { file: int8_matmul/mistakes/placeholders.py, fails: returns_float, says: "still returns `...`" }
  - { file: int8_matmul/mistakes/acc_placeholder.py, fails: returns_float, says: "`acc` is still `...`" }
  - { file: int8_matmul/mistakes/no_scale.py, fails: returns_float, says: "still in units of the two scales" }
  - { file: int8_matmul/mistakes/sw_column.py, fails: returns_float, says: "`sw.T`" }
  - { file: int8_matmul/mistakes/int8_acc.py, fails: no_wrap, says: "wrapped around" }
  - { file: int8_matmul/mistakes/float_matmul.py, fails: rescaled, says: "nothing was quantized" }
  - { file: int8_matmul/mistakes/only_sx.py, fails: rescaled, says: "never uses `sw`" }
  - { file: int8_matmul/mistakes/only_sw.py, fails: rescaled, says: "never uses `sx`" }
  - { file: int8_matmul/mistakes/divided.py, fails: rescaled, says: "You divided by the scales" }
```

### Split the outliers off: LLM.int8()

```challenge
id: mixed_matmul
file: outliers.py
minutes: 30
start: { from: int8_matmul }
setup:
  - say: "LLM.int8()'s fix: any feature column where some token goes past a threshold, 6.0 in the paper, skips int8 and multiplies in full precision. Everything else stays int8, and the two partial products add up."
    type_file: mixed_matmul/stub.py
    pause: 2.4
  - say: "Your turn: build the mask over feature columns, then the two-part matmul."
    pause: 1.0
tests: mixed_matmul/tests.py
solution: mixed_matmul/solution.py
alternatives:
  - mixed_matmul/alternatives/amax_mask.py
  - mixed_matmul/alternatives/f_linear.py
hints:
  nudge: "A matmul is a sum over features, so you can split the features into two sets, multiply each set separately, and add. Which features belong in which set?"
  concept: "`X.abs() > threshold` is a [4, 6] grid of bools. Reduce it over the tokens (`dim=0`) with `any`, and you get one flag per feature. `X[:, big]` keeps only the flagged columns of X, and `W[:, big]` the same features of W, since W is [out, features]. `~big` flips the mask for the int8 part."
  skeleton: |
    big = (X.____() > threshold).___(dim=___)
    return X[:, ___] @ W[:, ___].T + int8_matmul(X[:, ____], W[:, ____])
  solution: |
    big = (X.abs() > threshold).any(dim=0)
    return X[:, big] @ W[:, big].T + int8_matmul(X[:, ~big], W[:, ~big])
on_solve: "Mean error drops 20×, from 0.024 to 0.0012. Two of six columns ran in full precision here; in a real 2048-wide model it's a handful, under 1% of the matmul. The catch is speed: splitting, gathering and adding two results costs time, so LLM.int8() saves memory without being faster than fp16."
mistakes:
  - { file: mixed_matmul/mistakes/placeholders.py, fails: splits_columns, says: "still returns `...`" }
  - { file: mixed_matmul/mistakes/big_placeholder.py, fails: splits_columns, says: "`big` is still `...`" }
  - { file: mixed_matmul/mistakes/per_token.py, fails: splits_columns, says: "one per token" }
  - { file: mixed_matmul/mistakes/w_rows.py, fails: splits_columns, says: "picks W's *rows*" }
  - { file: mixed_matmul/mistakes/no_abs.py, fails: splits_columns, says: "but not feature 2" }
  - { file: mixed_matmul/mistakes/all_tokens.py, fails: splits_columns, says: "`all`" }
  - { file: mixed_matmul/mistakes/no_outlier_part.py, fails: splits_columns, says: "never added back" }
  - { file: mixed_matmul/mistakes/all_int8.py, fails: splits_columns, says: "through int8 too" }
  - { file: mixed_matmul/mistakes/fixed_threshold.py, fails: uses_threshold, says: "Use the `threshold` argument" }
```

### Migrate the scale: SmoothQuant

```challenge
id: smooth
file: outliers.py
minutes: 30
start: { from: mixed_matmul }
setup:
  - say: "LLM.int8() keeps a slow fp16 path. SmoothQuant keeps everything int8 instead. Divide feature j of X by a factor s_j and multiply feature j of W by the same s_j, and X @ W.T doesn't change, but the outlier shrinks."
    type_file: smooth/stub.py
    pause: 2.6
  - say: "Your turn: write the factor and the two rescaled tensors. Alpha decides how much of the difficulty moves into the weights."
    pause: 1.0
tests: smooth/tests.py
solution: smooth/solution.py
alternatives:
  - smooth/alternatives/pow_method.py
  - smooth/alternatives/negative_power.py
hints:
  nudge: Each feature gets one factor. The activations should get smaller where they're big, and the weights pick up the difference. Which side divides and which multiplies?
  concept: "`X.abs().amax(dim=0)` is the largest |x| per feature, shape [6]. W is [out, features], so `W.abs().amax(dim=0)` is the largest |w| per feature, also [6]. s = max|X|^alpha / max|W|^(1 − alpha). `X / s` and `W * s` broadcast that [6] along the features of each, and the product (X / s) @ (W * s).T is exactly X @ W.T."
  skeleton: |
    s = X.___().____(dim=___) ** alpha / W.___().____(dim=___) ** (1 - _____)
    return X _ s, W _ s
  solution: |
    s = X.abs().amax(dim=0) ** alpha / W.abs().amax(dim=0) ** (1 - alpha)
    return X / s, W * s
on_solve: "Look at the new int8 X. Feature 2 still sets each token's scale, but it's no longer 20× bigger than the rest, so the other features spread over dozens of levels (29, −49, 74) instead of single digits. The error drops 9× with no fp16 path at all. Try alpha = 0.75 in the call: more of the difficulty moves into W, and here the error drops a little further. SmoothQuant's paper used 0.5 for most models and up to 0.85 for the ones with the worst outliers."
mistakes:
  - { file: smooth/mistakes/placeholders.py, fails: returns_pair, says: "still returns `...`" }
  - { file: smooth/mistakes/s_placeholder.py, fails: returns_pair, says: "`s` is still `...`" }
  - { file: smooth/mistakes/w_dim1.py, fails: returns_pair, says: "reduce over dim 0 for both" }
  - { file: smooth/mistakes/no_abs.py, fails: same_product, says: "is nan" }
  - { file: smooth/mistakes/divided_both.py, fails: same_product, says: "divided both" }
  - { file: smooth/mistakes/swapped.py, fails: migrates, says: "got bigger" }
  - { file: smooth/mistakes/fixed_alpha.py, fails: migrates, says: "Use the `alpha` argument" }
  - { file: smooth/mistakes/alpha_swapped.py, fails: migrates, says: "exponents are swapped" }
  - { file: smooth/mistakes/mean.py, fails: migrates, says: "the mean" }
```

### Scan a real model for outlier features

```replay
id: scan_outliers
intro: "Your 30M model is too small to grow outlier features, so this run uses Llama 3.2 1B in fp16 (2.5 GB). It hooks every Linear's input, counts hidden dimensions over 6.0, then compares three int8 schemes on perplexity. About 25 minutes on a 12 GB card. Watch a recording first."
log: replays/scan-outliers-1b.jsonl
speed: 40
recorded:
  hardware: RTX 3060 12 GB
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: Largest |x| entering each layer
  flag_above: 6
  x_labels: [layers.0, layers.15]
  caption: "From layer 2 on, a few hidden dimensions sit far above 6 in every layer. The scan counts 6 dimensions that pass 6.0 in at least a quarter of the layers. Those are the outlier features that per-tensor int8 can't handle."
summary:
  - { label: Outlier dimensions, value: "6 of 2048" }
  - { label: Naive W8A8, value: "9.76 → 12.41 (+27.2%)" }
  - { label: LLM.int8(), value: "9.76 → 9.79 (+0.3%)" }
  - { label: SmoothQuant W8A8, value: "9.76 → 9.82 (+0.6%)" }
run_yourself: |
  # on your machine or Colab, from the course repo
  python scan_outliers.py --model meta-llama/Llama-3.2-1B --threshold 6.0
  python w8a8_eval.py --model meta-llama/Llama-3.2-1B \
    --schemes naive,llm_int8,smoothquant --alpha 0.5 --eval-samples 2000
```

### Record your metrics

```metrics
id: record
intro: The perplexity change for SmoothQuant W8A8 against fp16, and how many outlier dimensions your scan found.
```

## Hints

- Hook the *inputs* of each `nn.Linear` (`register_forward_hook` gives you them as `inputs[0]`), not the outputs. The inputs are what gets quantized to int8.
- Count a dimension as an outlier the way LLM.int8() does: |x| ≥ 6 in at least 25% of the layers and in at least 6% of the token positions. A single spike in one token isn't a feature.
- Naive W8A8 here means one int8 scale for the whole activation tensor. Per-token scales, as in your `int8_matmul`, do better but still lose the small features in every token.
- SmoothQuant's factors come from a calibration pass (a few hundred samples), computed once, and are folded into the previous LayerNorm's weight, so inference pays nothing for them.

## Done when

- [ ] Mixed precision beats plain int8 on the lesson's tensors {#mixed auto=challenge:mixed_matmul}
- [ ] SmoothQuant keeps X @ W.T and shrinks the outlier {#smooth auto=challenge:smooth}
- [ ] Perplexity change and outlier count recorded {#metrics_recorded auto=metrics}
