---
gist: In big models a few activation features are 20 to 100 times larger than the rest in almost every token, which ruins int8 matmuls; you can split them off into full precision (LLM.int8()) or move their size into the weights (SmoothQuant).
minutes: 17
---

## What this lesson is for {#why}
> So far you quantized only weights. To make the matmul itself run in int8 you also have to quantize the activations, and in large models a few huge activation values make that hard.

In 3.1 and 3.2 the weights were stored in 8 or 4 bits, but every matmul still ran in fp16: the weights were turned back into floats first. That saves memory, not arithmetic.

A matmul has two inputs: the weights, and the **activations** flowing through the model. GPUs can multiply int8 by int8 roughly twice as fast as fp16 by fp16. To use that, both inputs have to be int8. This is called **W8A8**: 8-bit weights, 8-bit activations.

Weights are calm. Activations are not. In models of a few billion parameters and up, a handful of activation **features** carry values 20 to 100 times bigger than the rest, in almost every token. They wreck int8 the same way 3.1's outlier row did, but worse, because they show up everywhere.

In this lesson you:

1. write an int8 × int8 matmul, as LLM.int8() does it;
2. watch one outlier feature ruin it;
3. fix it by splitting the outliers off (**LLM.int8()**);
4. fix it again by moving their size into the weights (**SmoothQuant**);
5. watch a scan of a real 1B model for outlier features.

The recording in 3.1 already showed one bad row. Lesson 3.4 then goes back to weights and asks how to round them more cleverly.

## What you need from before {#before}
> A Linear layer computes X @ W.T, absmax gives one scale per row, and `amax(dim=…)` takes the largest value along one dimension.

**The Linear layer's matmul.** Activations `X` have one row per token and one column per **feature** (hidden dimension). The weight `W` has one row per output and one column per feature, as `nn.Linear` stores it. The layer computes `X @ W.T`:

```
X      [tokens, features]   [4, 6]
W      [out, features]      [3, 6]
X@W.T  [tokens, out]        [4, 3]
```

Each output `[i, j]` is a sum over features: token i's feature k times output j's weight for feature k, added up for every k.

**Absmax per row (3.1).** scale = max|row| / 127, then q = round(row / scale). Error is at most half a step.

**Reductions with `dim` (phase 0).** `amax(dim=0)` goes down the columns and gives one value per column. `dim=1` goes across each row. `keepdim=True` keeps the result as a column or row so it broadcasts.

**Bool masks (phase 0).** `x > 6` gives a tensor of `True`/`False`, and `t[:, mask]` keeps only the columns where the mask is `True`.

## Two scales for one matmul {#vector-wise}
> Give every token its own scale and every output its own scale. Multiply the integers, then multiply the result by the two scales.

LLM.int8()'s **vector-wise** quantization quantizes `X` per row (one scale per token) and `W` per row (one scale per output). Why does that work for a matmul? Each original value is roughly its code times its scale, so

```
x_k × w_k ≈ (qx_k × sx) × (qw_k × sw)
          = sx × sw × qx_k × qw_k
```

Both scales are the same for every k in the sum, so they come outside: the output is sx × sw times the sum of integer products. Do the whole sum in integers, then multiply once at the end.

A small check by hand. One token `x = [1.27, −0.63]` and one weight row `w = [0.2, 0.5]`:

| | values | scale | codes |
|---|---|---|---|
| x | 1.27, −0.63 | 1.27 / 127 = 0.01 | 127, −63 |
| w | 0.2, 0.5 | 0.5 / 127 ≈ 0.003937 | 51, 127 |

```
integer sum:
127×51 + (−63)×127
= 6477 − 8001 = −1524

rescaled:
−1524 × 0.01 × 0.003937
≈ −0.0600

exact:
1.27×0.2 + (−0.63)×0.5
= −0.061
```

Close, and every multiply-add was on whole numbers.

**Why int32 for the sum.** 127 × 51 is already 6,477, far past int8's 127. Keep the sum in int8 and it wraps around, like Java's `byte`. Java protects you here: `byte * byte` is computed as an `int`. GPUs do the same, multiplying int8s and adding the products in int32. In PyTorch you have to ask for it by converting the codes to a wider integer type before the matmul. The tests spot wrapped sums and tell you.

**Lining up the scales.** The integer result has shape `[tokens, out]`. Token scales form a `[tokens, 1]` column, which matches its rows. Output scales also come out as a `[out, 1]` column, but outputs run along the result's *columns*, so they have to be laid out as a `[1, out]` row before multiplying. Get this wrong and PyTorch complains that the sizes don't match.

## What an outlier feature is {#outliers}
> An outlier feature is one hidden dimension that carries huge values, about 6 or more, in most layers and many tokens.

Dettmers and colleagues measured activations in models of different sizes. In small models, large values come and go. From about 6.7 billion parameters on, the same few hidden dimensions are large in almost every layer and most tokens, with values around 20 to 60 while the rest stay near 1. They called a dimension an **outlier feature** if its magnitude reaches 6 in at least a quarter of the layers and in at least 6% of the token positions.

The lesson's toy activations have this shape. Here is the largest |x| per feature over the four tokens:

| Feature | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| max \|x\| | 0.88 | 0.95 | 21.0 | 0.60 | 0.95 | 9.6 |
| Notes | | | about −20 in every token | | | large in 2 of 4 tokens |

Feature 2 is the classic case: huge and negative in every token. Feature 5 is big in only two tokens.

## Why one feature ruins every token {#why-it-hurts}
> Per-token absmax takes each token's scale from its biggest feature, so an outlier present in every token sets every token's scale.

Take token 2 of the lesson, `[0.35, 0.72, −19.2, 0.60, −0.44, 0.21]`. Its scale is 19.2 / 127 ≈ 0.151.

| x | x / scale | code | back | error |
|---|---|---|---|---|
| 0.35 | 2.3 | 2 | 0.302 | 0.048 |
| 0.72 | 4.8 | 5 | 0.756 | 0.036 |
| −19.2 | −127 | −127 | −19.2 | 0 |
| 0.60 | 4.0 | 4 | 0.605 | 0.005 |
| −0.44 | −2.9 | −3 | −0.454 | 0.014 |
| 0.21 | 1.4 | 1 | 0.151 | 0.059 |

The five ordinary features share codes between −3 and 5, out of 255 available. Without the outlier, the scale would be 0.72 / 127 ≈ 0.0057, and no error would exceed 0.003.

3.1's per-row scale contained a *weight* outlier to its own row. An *activation* outlier sits in the same feature of every token, so per-token scales can't contain it. Every row of the int8 `X` ends up with −127 in column 2 and single digits everywhere else.

The slider card lets you feel this: find how many bits one such token needs before its small features come out right.

## Why not just clip them {#no-clipping}
> The outliers carry information the model relies on, so cutting them down to size breaks the model.

The obvious fix is to clip activations to ±6 and quantize the rest. Dettmers et al. tried removing the outlier dimensions, and perplexity got much worse, even though they're a tiny fraction of all features. The model has learned to use them. Any fix has to carry them through accurately.

## Fix 1: split them off (LLM.int8()) {#decomposition}
> A matmul is a sum over features, so you can do the outlier features in full precision, the rest in int8, and add the two results.

Since each output is a sum over features, you can split the features into two groups, sum each group separately and add:

```
full sum
= sum over outlier features
+ sum over ordinary features
```

LLM.int8() does the outlier part in fp16 and the ordinary part with the int8 matmul. In a real 2048-wide model only a handful of features are outliers, under 1% of the work, so almost everything stays int8.

To build the split you need one **True/False flag per feature**: is it an outlier? The rules that the tests check:

- **One flag per feature, not per token.** The mask has 6 entries, one per column of `X`. Reducing the wrong dimension gives 4 entries, one per token, and the column selection fails.
- **Any token is enough.** Feature 5 is big in only two tokens, but in those two it sets the scale. A feature counts as an outlier if *any* token passes the threshold, not *every* token.
- **Size, not sign.** Feature 2 is about −20. Compare its magnitude with the threshold, or the most extreme feature slips through.
- **W's features are columns too.** `W` is `[out, features]`, so you pick the same columns from `W` as from `X`, not rows.
- **Use the threshold you're given.** The tests call your function with a different threshold.

On the lesson's tensors this drops the mean output error from 0.024 to 0.0012. The catch is speed: finding, gathering and adding two separate results costs time, so LLM.int8() saves memory but isn't faster than fp16.

## Fix 2: move the size into the weights (SmoothQuant) {#smoothquant}
> Divide feature k of X by a factor s_k and multiply feature k of W by the same s_k. The product doesn't change, but the outlier shrinks.

Look at one term of the sum again:

```
x_k × w_k = (x_k / s_k) × (w_k × s_k)
```

Dividing one side and multiplying the other by the same number changes nothing in the exact product. But it changes what each side looks like to the quantizer. Pick a large s_k for the outlier feature and its activations shrink, while its weights grow. Weights had no outliers to begin with, so they have room.

SmoothQuant chooses, for each feature k:

```
s_k = max|X_k|^α / max|W_k|^(1−α)
```

where max|X_k| is the largest activation in feature k over the tokens, and max|W_k| the largest weight for feature k over the outputs. With α = 0.5 both powers are square roots, and afterwards both sides have the *same* maximum for that feature.

For the lesson's feature 2, max|X| = 21.0 and max|W| = 0.07:

```
s = √21 / √0.07
  = 4.58 / 0.265 ≈ 17.3

new max |X| = 21 / 17.3   ≈ 1.21
new max |W| = 0.07 × 17.3 ≈ 1.21
```

The activation went from 20 times bigger than its neighbours to merely the biggest. Features without outliers get factors near 1 and barely change.

**α** decides how much moves. α = 0 changes nothing, α = 1 moves everything into the weights, which then become the problem. 0.5 works for most models; models with the worst outliers use more.

Mistakes the tests check: dividing and multiplying the wrong way round (the outlier gets bigger), dividing both sides (the product changes), swapping the two exponents, using the mean instead of the max, ignoring α, and forgetting the magnitude. Feature 2 is all negative, so its plain max is −17.8, and a negative number to the power 0.5 is `nan`. W's features are its columns, so its max also runs over dim 0.

On the lesson's tensors, smoothing cuts int8 error about 9× with no fp16 path at all. In a real model the division of X is folded into the LayerNorm that produces X, so inference pays nothing for it.

## Running it on your GPU {#gpu}
> Your 30M model is too small to grow outlier features, so the run scans Llama 3.2 1B and compares three int8 schemes.

The script hooks the input of every Linear layer, feeds a few hundred samples through Llama 3.2 1B in fp16 (2.5 GB), and counts hidden dimensions above 6. Then it measures perplexity for three W8A8 schemes. About 25 minutes on a 12 GB card. The lesson shows an **illustrative** recording first: sample data written for the course, not a real run.

A healthy scan looks like this:

- layers 0 and 1 stay below 6;
- from layer 2 on, a few dimensions sit far above 6 in every layer, up to about 60;
- 6 of the 2048 dimensions count as outlier features.

And the perplexity comparison:

| Scheme | Perplexity | Change |
|---|---|---|
| fp16 | 9.76 | |
| naive W8A8, one scale per tensor | 12.41 | +27.2% |
| LLM.int8(), threshold 6 | 9.79 | +0.3% |
| SmoothQuant W8A8, α = 0.5 | 9.82 | +0.6% |

You record SmoothQuant's perplexity change (should stay under 2%) and the number of outlier dimensions your scan found (at least 1).

## Words you'll see {#glossary}
> The terms in this lesson's cards, challenges and metrics.

- **Activations**: the values flowing between layers, different for every input. Shape `[tokens, features]` at a Linear's input.
- **Feature (hidden dimension, channel)**: one column of the activations. A 2048-wide model has 2048.
- **W8A8**: 8-bit weights and 8-bit activations, so the matmul itself runs in int8.
- **Weight-only quantization**: only the weights are stored small; the matmul runs in floats.
- **Vector-wise quantization**: one scale per token of X and one per output of W.
- **Per-token / per-tensor scale**: one scale for each token's row, or one for the whole tensor.
- **int32 accumulation**: adding int8 × int8 products in a 32-bit integer so the sum doesn't wrap.
- **Outlier feature**: a hidden dimension with values of 6 or more in many layers and tokens.
- **Threshold**: the size above which a feature counts as an outlier, 6.0 in LLM.int8().
- **Mask**: a True/False tensor that picks columns.
- **Mixed-precision decomposition**: LLM.int8()'s split into an fp16 part for outliers and an int8 part for the rest.
- **LLM.int8()**: the paper and method that found outlier features and introduced the split.
- **SmoothQuant**: the method that migrates activation size into the weights.
- **Smoothing factor s**: the per-feature number X is divided by and W multiplied by.
- **α (alpha)**: how much of the size moves from activations to weights, between 0 and 1.
- **Migration**: moving quantization difficulty from one operand to the other.
- **Calibration data**: a small set of sample text used to measure activations before quantizing.
- **nan**: "not a number", what a negative number to a fractional power gives.
- **Perplexity change (ppl delta)**: the quality cost in % against fp16.
