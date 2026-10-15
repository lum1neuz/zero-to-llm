---
gist: Rounding each weight on its own ignores what the layer's inputs look like; GPTQ uses real inputs to round one column at a time and pushes each rounding error onto weights that can cancel it.
minutes: 17
---

## What this lesson is for {#why}
> At 4 bits, plain rounding costs Llama 3.1 8B almost 9% in perplexity. GPTQ halves that by using real inputs to decide how to round.

Since 3.1 every method has rounded each weight to its nearest level, on its own. That's called **round-to-nearest**, or **RTN**. It's fast, needs no data, and at 8 bits it's good enough. At 4 bits it starts to hurt. In this lesson's recording, RTN with groups of 128 makes Llama 3.1 8B's perplexity 8.8% worse.

**GPTQ** keeps the same 4-bit format, the same levels and the same scales. It only chooses *differently* which level each weight gets, and in the recording that brings the cost down to 4.5%. It does this by looking at **calibration data**: a small batch of real text pushed through the model, so it knows what each layer's inputs actually look like.

You will:

1. build the matrix that describes those inputs, the **Hessian**, including the small fix that keeps it usable;
2. write GPTQ's column-by-column loop on a 4 × 8 layer and compare it with RTN;
3. watch GPTQ quantize Llama 3.1 8B on a 12 GB card, one block at a time.

3.5 then shows two cheaper ideas, AWQ and NF4, and 3.6 the formats llama.cpp ships. Many 4-bit models you download today were made with GPTQ.

## What you need from before {#before}
> A layer computes X @ W.T, RTN rounds each weight to its row's grid, and a transpose swaps rows and columns.

**The layer.** Inputs `X` are `[tokens, features]` and weights `W` are `[out, features]`. The output `X @ W.T` is `[tokens, out]`. Each output adds up input × weight over the features (3.3).

**RTN with a row scale (3.1, 3.2).** step = max|row| / 7 at int4. Each weight goes to the nearest multiple of the step, clamped to ±7 steps.

**Transpose.** `X.T` swaps rows and columns, so a `[64, 8]` tensor becomes `[8, 64]`. Matrix shapes multiply like `[a, b] @ [b, c] → [a, c]`; the inner sizes have to match and disappear.

**Outputs, not weights.** 3.3 already measured error on the matmul's *output*. That's the measure here too.

## Weight error isn't what matters {#objective}
> The model only ever sees a layer's output, X @ W.T. A rounding choice that's worse for the weights can be better for the output.

Imagine two features that are always equal: in every token, feature 1 has the same value as feature 0. Then the layer only ever sees the sum w0 + w1. Say both weights are 0.40 and the grid step is 0.25, so a weight can be 0.25 or 0.50.

| | w0 | w1 | w0 + w1 | Output error |
|---|---|---|---|---|
| Original | 0.40 | 0.40 | 0.80 | |
| RTN | 0.50 | 0.50 | 1.00 | 0.20 |
| Round w0, then fix w1 | 0.50 | 0.25 | 0.75 | 0.05 |

RTN rounds both up, because 0.40 is nearer 0.50. Each weight is as close as it can be, and the output is off by 0.20 on every token.

The second row rounds w0 first. It moved up by 0.10. Because feature 1 always equals feature 0, taking 0.10 off w1 cancels that move exactly: w1 becomes 0.30, which then rounds to 0.25. The sum is 0.75, four times closer. Each weight's own error got *bigger*, and the output got better.

That's GPTQ's whole idea. Real features aren't exact copies, but they're strongly correlated, so there's almost always a later weight that can absorb some of an earlier weight's error.

## Correlated inputs {#correlated}
> Features in real activations move together, which is what gives GPTQ room to cancel errors.

Activations aren't random. A few underlying patterns in the text show up in many features at once. The lesson's calibration batch imitates that: 64 tokens whose 8 features are mixtures of 3 hidden causes, plus a little noise. Feature 7 never fires in this batch, which happens in real models too and matters in a moment.

If features were completely independent, there would be nothing to cancel against, and GPTQ would give the same answer as RTN.

## The Hessian: a table of how features move together {#hessian}
> H = 2XᵀX / n has one row and one column per feature. Entry [i, j] sums feature i times feature j over all tokens.

GPTQ needs to know, for any change to a row of weights, how much the output changes. That information sits in one matrix built from the inputs:

```
H = 2 · Xᵀ X / n
X     [n tokens, features]  [64, 8]
Xᵀ    [features, n tokens]  [8, 64]
XᵀX   [features, features]  [8, 8]
```

The transpose goes on the *left*. `X @ X.T` would be `[64, 64]`, one row per token, which describes tokens, not features. That's a classic mistake the tests catch by shape.

A tiny one by hand. Three tokens, two features:

```
X = [[1, 1],
     [2, 2],
     [1, 0]]

XᵀX = [[6, 5],
       [5, 5]]
```

[0, 0] is 1·1 + 2·2 + 1·1 = 6: how strongly feature 0 fires. [0, 1] is 1·1 + 2·2 + 1·0 = 5: how much features 0 and 1 fire *together*. A big off-diagonal entry means the two features are correlated, so one can stand in for the other.

**Why it describes output error.** Change a row of weights by δ. Each token's output changes by x · δ, and the total squared error is δᵀ(XᵀX)δ. With the tiny X above:

| Change δ | Output change per token | Squared error |
|---|---|---|
| [+0.1, −0.1] | 0, 0, 0.1 | 0.01 |
| [+0.1, +0.1] | 0.2, 0.4, 0.1 | 0.21 |

Both changes are the same size for the weights. The first is nearly invisible to the inputs, the second isn't. H tells GPTQ which is which. In maths, the matrix of second derivatives of an error is called its **Hessian**, hence the name. The 2 / n in front only scales H, and GPTQ's result doesn't depend on that.

## Damping: keeping H invertible {#damping}
> A feature that never fires leaves a row and column of zeros in H, and then H has no inverse. GPTQ adds a little to every diagonal entry.

GPTQ needs the **inverse** of H, written H⁻¹. It plays the role of 1 / x for matrices. And just as 1 / 0 doesn't exist, a matrix with a row of zeros has no inverse.

Feature 7 never fires, so its row and column in XᵀX are all zero. GPTQ's fix is **damping**: add a small number to every diagonal entry, H[i, i], and leave the rest alone. The amount is relative to H's own size:

```
damp = percdamp × (mean of H's diagonal)
percdamp = 0.01, so 1%
```

In the lesson's batch the diagonal averages about 7, so damp is about 0.07, and H[7, 7] goes from 0 to about 0.07. Features that do fire hardly notice.

The tests check the classic ways to get this wrong:

| Mistake | What goes wrong |
|---|---|
| no damping | H[7, 7] stays 0, no inverse |
| add damp to every entry | changes how features relate |
| a fixed 0.01 | too small when X is large |
| scale each diagonal entry by 1.01 | 0 × 1.01 is still 0 |
| mean of *all* entries | not the diagonal's mean |
| ignore the `percdamp` argument | the tests pass 0.1 too |

## The column loop {#loop}
> Quantize one column of W at a time. After each, spread its rounding error over the columns not yet quantized, using numbers derived from H⁻¹.

W is `[4, 8]`: 4 output rows, 8 feature columns. GPTQ walks the columns left to right. For column j:

1. **Round it.** Column j holds one weight per row, shape `[4]`. Each goes to the nearest level on its row's int4 grid, exactly like 3.2, and is kept as a float on the grid.
2. **Measure the error.** How far each of the 4 weights moved, divided by one number from H⁻¹'s factor (below). Shape `[4]`, one per row.
3. **Correct the rest.** Subtract a correction from columns j onwards. The correction is the outer product of the 4 errors with one row of numbers, one per remaining column. Shape `[4, 8 − j]`.

Two shape points come up in the tests. The error is one value per row and the weights come in one value per remaining column, so you need the error as a `[4, 1]` column to multiply out a `[4, 8 − j]` block. And the result you store must be on the grid in the weights' units, not the integer codes.

**Where the numbers come from.** The correction for column j uses row j of a matrix called U. U is the **Cholesky factor** of H⁻¹: an upper-triangular matrix (zeros below the diagonal) with H⁻¹ = UᵀU, a kind of square root for matrices. Row j of U, from column j on, says how much of column j's error each later column should absorb. Its first entry, U[j, j], scales the error. It has the bookkeeping for "columns before j are already fixed" built in. The rows of plain H⁻¹ don't: they're right for column 0 and go stale after that.

You don't compute U yourself. PyTorch does it with `torch.linalg`, which mini-torch doesn't have, so the setup types two short helper functions that compute the inverse and the Cholesky factor the long way.

Classic mistakes: no correction at all (that's RTN again), not dividing by U[j, j], adding the correction instead of subtracting it, and using rows of H or of plain H⁻¹ instead of U.

## Trading weight error for output error {#tradeoff}
> On the lesson's layer GPTQ's weight error is higher than RTN's, and its output error is about 7× lower.

| | Mean weight error | Output error |
|---|---|---|
| RTN | 0.0127 | 0.0092 |
| GPTQ | 0.0158 | 0.0013 |

That's the two-weight example at full size: weights drift in directions the inputs can't see.

**Calibration data matters.** H is built from the calibration inputs, so GPTQ protects the output on inputs like those. The usual set is 128 samples of 2048 tokens of generic text. Calibrate on text unlike what you'll serve, and the compensation goes to waste. For the Mongolian model in phase 4, calibrate on Mongolian.

## GPTQ on a real model {#at-scale}
> The real model is quantized one transformer block at a time, so an 8B model that doesn't fit in 12 GB still quantizes on a 12 GB card.

Hessians are big. H has one row and column per input feature, so a Linear with 4096 inputs has a 4096 × 4096 H:

```
4096 × 4096 × 4 bytes
= 67,108,864 bytes ≈ 67 MB
```

GPTQ therefore works block by block: run the calibration samples through the blocks quantized so far, record each Linear's inputs in the current block, build their Hessians, quantize, replace the weights, free the memory and move on. Each block sees inputs that already passed through quantized blocks, so it corrects for their errors too.

The real implementation adds one speedup and two accuracy tricks:

- **lazy batches** (speed): update 128 columns at a time before touching the rest of the matrix, which suits GPU memory;
- **group scales** (accuracy): one scale per 128 weights, as in 3.2;
- **act-order** (accuracy): quantize the columns with the largest H diagonal first, while there are the most columns left to absorb their error.

The two accuracy tricks cost a little speed.

## Running it on your GPU {#gpu}
> The run quantizes Llama 3.1 8B with GPTQ int4, groups of 128, in about 20 minutes, then compares perplexity with fp16 and RTN.

The lesson shows an **illustrative** recording first: sample data written for the course, not a real run. A healthy run looks like this:

- the model loads into CPU memory (16.1 GB in fp16) and only one block at a time goes to the GPU;
- 128 calibration samples of 2048 tokens;
- one line per block, 32 in all, with an output error that grows with depth; the last two blocks stand out, as they usually do for Llama;
- about 21 minutes in total, peak GPU memory about 6.8 GB;
- a 5.7 GB folder: the Linear layers in int4, embeddings and the output layer in fp16.

The fp16 baseline doesn't fit in 12 GB, so its perplexity eval offloads layers to CPU memory and is slow (almost an hour). Results in the recording:

| Model | Perplexity | Change |
|---|---|---|
| fp16 | 6.24 | |
| RTN int4 g128 | 6.79 | +8.8% |
| GPTQ int4 g128 | 6.52 | +4.5% |

You record GPTQ's perplexity change (under 6% passes) and the minutes it took (under 60). If quantization fails with a Cholesky error or the output turns into NaN, raise the damping from 0.01 to 0.1.

## Words you'll see {#glossary}
> The terms in this lesson's cards, challenges and metrics.

- **RTN (round-to-nearest)**: rounding every weight to its nearest level on its own.
- **GPTQ**: quantizing column by column and compensating each column's error in the columns after it.
- **Calibration data**: a small sample of real text used to see what each layer's inputs look like.
- **Output error**: how much X @ W.T changes, the measure GPTQ minimises.
- **Weight error**: how far each weight moved. GPTQ lets it grow.
- **Correlated features**: features that tend to be large or small together.
- **Hessian (H)**: 2XᵀX / n, a `[features, features]` table of how features move together; it measures output error for any weight change.
- **Diagonal**: the entries H[i, i] from top-left to bottom-right.
- **Inverse (H⁻¹)**: the matrix that undoes H, like 1 / x for numbers. A zero row means it doesn't exist.
- **Damping, percdamp**: adding a small amount (percdamp × the diagonal's mean, usually 1%) to the diagonal so H can be inverted.
- **Dead feature**: a feature that is zero for every calibration token.
- **Cholesky factor (U)**: an upper-triangular matrix with H⁻¹ = UᵀU. Its rows drive GPTQ's corrections.
- **Upper-triangular**: zeros everywhere below the diagonal.
- **Outer product**: a column times a row, giving a full grid, here `[rows, columns left]`.
- **Compensation**: the correction added to the later columns.
- **Lazy batch updates**: applying corrections 128 columns at a time, for speed.
- **Group size (g128)**: one scale per 128 weights.
- **Act-order (desc_act)**: quantizing the columns with the largest diagonal first.
- **Block**: one transformer layer (attention plus MLP) with its 7 Linear layers.
- **CPU offload**: keeping some layers in normal RAM when the GPU is too small. Slow but it works.
