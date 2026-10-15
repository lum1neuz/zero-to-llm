---
gist: AWQ gives finer steps to the weights that big activations depend on, and NF4 puts its 16 levels where weights actually are; both make 4 bits go further without GPTQ's heavy machinery.
minutes: 17
---

## What this lesson is for {#why}
> Two more ways to spend 4 bits well. AWQ decides which weights deserve finer steps; NF4 changes where the 16 levels sit.

GPTQ (3.4) rounds first and then repairs the damage with a Hessian. This lesson's two methods make the rounding itself land better, and both are much simpler to run.

- **AWQ** (activation-aware weight quantization) notices that a few input channels carry much larger activations than the rest. Errors in the weights of those channels hurt the output most, so AWQ quietly gives them finer steps.
- **NF4** (4-bit NormalFloat) keeps one scale per block like 3.2, but replaces int4's 15 evenly spaced levels with 16 levels placed where weights are most common.

NF4 matters for the rest of the course. It's the format QLoRA trains on top of, and in phase 4 you'll load your base model in NF4 and train LoRA adapters on it.

You'll write AWQ's scaling and NF4's lookup on small tensors, then watch both run on Llama 3.1 8B and compare them with GPTQ.

## What you need from before {#before}
> int4 absmax per row, output error from 3.4, and SmoothQuant's divide-and-multiply trick from 3.3.

**int4 absmax (3.2).** step = max|row| / 7, codes −7…7. The row's largest weight decides the step for everyone else in it.

**Output error (3.4).** What matters is how much `X @ W.T` changes, not how far each weight moved.

**The divide-and-multiply trick (3.3).** For each input channel k, x_k × w_k = (x_k / s) × (w_k × s). SmoothQuant used it to shrink activation outliers. AWQ uses the same identity, but for the weights' benefit.

**Lookups by index (phase 0).** `table[idx]` with a tensor of integer positions returns the entries at those positions, in idx's shape.

## Not every weight matters equally {#salient}
> An error in weight w_k reaches the output multiplied by its input x_k, so weights on channels with big activations matter more.

One output is a sum of x_k × w_k over the input channels. Suppose quantization moves one weight by 0.01. The output moves by 0.01 × x_k.

| Channel's typical \|x\| | Weight error | Output error |
|---|---|---|
| 1 | 0.01 | 0.01 |
| 6 | 0.01 | 0.06 |

The same weight error does six times the damage on the busy channel. AWQ's authors found that protecting about 1% of the weights, the ones on the channels with the largest activations, recovers most of the quality lost at 4 bits. Picking them by the weights' own size doesn't work nearly as well. You have to look at the activations, hence "activation-aware".

These important channels are called **salient**. In the lesson's toy layer, channels 2 and 6 have activations about 6× and 3× bigger than the rest.

Keeping salient weights in fp16 would work, but a matrix mixing formats is awkward to store and slow to multiply. AWQ finds a way to protect them with every weight still in int4.

## Stretch, quantize, shrink {#scaling}
> Multiply a salient weight column by s before quantizing and divide by s afterwards. As long as it doesn't become the row's largest weight, its error shrinks by about s.

The step size is set by the row's largest weight. Take a row whose largest weight is 0.70, so the int4 step is 0.70 / 7 = 0.1, and a salient weight of 0.14 in it.

| Factor s | Stretched | Step | Rounded | Shrunk back | Error |
|---|---|---|---|---|---|
| 1 (plain) | 0.14 | 0.10 | 0.10 | 0.10 | 0.04 |
| 2 | 0.28 | 0.10 | 0.30 | 0.15 | 0.01 |

Stretching made the weight bigger relative to an unchanged step. After shrinking back, its worst possible error drops from half a step (0.05) to half a step divided by s (0.025), and here it lands at 0.01. Dividing the matching activation by s at run time keeps the output the same, exactly as in SmoothQuant.

**Too much stretch backfires.** With s = 6 the weight becomes 0.84, bigger than 0.70. Now *it* sets the row's step, 0.84 / 7 = 0.12, and every other weight in the row gets coarser steps than before. There's a sweet spot.

The order matters, and the tests check it: stretch, then quantize, then shrink back. Forget to shrink back and the salient columns come out s times too big. Shrink before quantizing and the important weights get the *coarsest* steps.

## Choosing the factors with α {#alpha}
> AWQ sets s for each channel to its average |x| raised to a power α, then tries a few values of α and keeps the best.

AWQ ties every channel's factor to how big that channel's activations are:

```
s_k = (mean |x_k| over tokens)^α
```

Channels with big activations get big factors. **α** controls how strongly: α = 0 makes every factor 1, which is plain RTN, and larger α stretches harder. AWQ tries a grid of α values per layer and keeps whichever gives the lowest output error on calibration data.

On the lesson's layer:

| α | Output error |
|---|---|
| 0 (RTN) | 0.020 |
| 0.25 | 0.014 |
| 0.5 | 0.0047 |
| 0.75 | 0.015 |
| 1.0 | 0.013 |

0.5 wins by more than 4× over RTN. Past it, the salient columns start to set their rows' steps, as in the s = 6 example.

Details the tests check:

- **The activations, not the weights.** The factors come from X.
- **The mean, not the max.** One spiky token shouldn't decide the factor. The max is SmoothQuant's choice, for a different purpose.
- **Average over the tokens.** One factor per channel, so the mean runs down X's columns, giving 8 numbers that line up with W's 8 columns.
- **Magnitudes.** A channel can average to a negative number, and a negative number to the power 0.5 is `nan`.
- **Use α as given.** The tests try several values.

## AWQ next to GPTQ {#vs-gptq}
> AWQ fits one number per channel and one α per layer, so it's fast and hard to overfit; GPTQ fits much more.

| | GPTQ | AWQ |
|---|---|---|
| Uses calibration data for | a full Hessian per layer | average \|x\| per channel |
| Changes weights by | column-by-column corrections | a per-channel stretch |
| Search | none, solved directly | a small grid over α |
| Overfitting risk | higher | low |

Because AWQ learns so little from the calibration set, it holds up when the text you serve differs from the text you calibrated on. In a real model the 1 / s on the activations is folded into the previous layer, so inference costs nothing extra.

## Evenly spaced levels waste codes {#levels}
> Trained weights are roughly bell-shaped: most sit near zero and few near the edges. int4's evenly spaced levels spend as many codes on the empty edges as on the crowded middle.

If you plot a trained layer's weights, you get a bell curve, a **normal distribution**. Scale a block of 64 by its absmax, so it runs from −1 to 1, and the crowd is near 0:

```
           ▄▆█▆▄
         ▂██████▂
      ▁▃▆████████▆▃▁
 ▁▁▂▄▆██████████████▆▄▂▁▁
-1          0          +1
```

int4 absmax puts levels at multiples of 1/7 ≈ 0.143 everywhere. The middle, where most weights are, gets the same spacing as the nearly empty edges. And symmetric int4 leaves one of its 16 codes, −8, unused.

## NF4: a 16-level palette {#nf4}
> NF4's 16 levels sit at quantiles of a normal distribution, so each level gets about the same share of the weights.

Think of a GIF image. Instead of storing each pixel's full colour, a GIF picks a **palette** of 256 colours suited to that picture and stores each pixel as a palette index. NF4 does the same with a 16-entry palette of values in [−1, 1]:

| Code | Level | Code | Level |
|---|---|---|---|
| 0 | −1.0000 | 8 | 0.0796 |
| 1 | −0.6962 | 9 | 0.1609 |
| 2 | −0.5251 | 10 | 0.2461 |
| 3 | −0.3949 | 11 | 0.3379 |
| 4 | −0.2844 | 12 | 0.4407 |
| 5 | −0.1848 | 13 | 0.5626 |
| 6 | −0.0911 | 14 | 0.7230 |
| 7 | 0.0000 | 15 | 1.0000 |

The levels are **quantiles**: cut a bell curve into slices that each hold the same share of values, and put a level in each. Near zero, where weights are dense, levels are about 0.08 apart; near the edges, about 0.3. Zero is a level of its own.

To quantize a block:

1. find the block's absmax, as in 3.2;
2. divide the block by it, so it lies in [−1, 1];
3. for each weight, find the **nearest** level and store its code, 0 to 15.

Take a normalised weight of 0.20. It sits between 0.1609 (0.039 away) and 0.2461 (0.046 away), so it gets code 9 and comes back as 0.1609. int4 would turn it into 1/7 ≈ 0.143, 0.057 away. Not every value comes out better, but on bell-shaped weights the average error is lower, about 30% lower in the lesson's challenge at the same 4 bits.

## Finding the nearest level with shapes {#shapes}
> Compare every weight with all 16 levels at once by adding a dimension, then take the position of the smallest distance.

Levels aren't evenly spaced, so you can't round. You measure distances:

```
normalised   [blocks, 64]
add a dim    [blocks, 64, 1]
levels       [16]
gaps         [blocks, 64, 16]
nearest      [blocks, 64]
```

Broadcasting `[blocks, 64, 1]` against `[16]` gives every weight a row of 16 gaps. Make them distances (drop the sign), then take the *position* of the smallest along the last dimension. That's an argmin, which returns indices, not values.

The tests check:

| Mistake | What you get |
|---|---|
| no extra dimension | shapes don't match |
| reduce the wrong dimension | one answer per level, not per weight |
| skip normalising | every weight lands near zero |
| argmin of signed gaps | every weight gets code 15 |
| argmax | the furthest level |
| round like int4 | wrong codes, levels aren't even |

On the way back: look the levels up by code, multiply by the block's absmax and flatten to one row. One trap: codes are stored as `uint8` to save space, and PyTorch reads a `uint8` (or `bool`) tensor used as an index as a **mask**, not as positions. Convert codes to a normal integer type before using them as an index.

## Double quantization {#double-quant}
> The per-block absmax values cost bits too. QLoRA quantizes them as well, which cuts their cost to about a quarter.

Each block of 64 weights needs its absmax. Stored as fp32, that's 32 bits shared by 64 weights:

```
32 / 64 = 0.5 bits per weight
```

So plain NF4 costs 4.5 bits per weight, not 4. **Double quantization** treats those absmax values like weights. It stores each one in 8 bits, and keeps one fp32 constant for every 256 blocks to decode them. The calc card has you add up the two shares; the total comes to about a quarter of the 0.5 bits.

**Not everything is quantized.** Llama 3.1 8B has a 128k-token vocabulary, so its embedding matrix and its output layer (`lm_head`) hold about 1 billion weights between them. bitsandbytes leaves both in bf16, 2 bytes per weight. That's why the NF4 model takes about 5.7 GB in memory, not 4.

**bf16** is another 16-bit float format: the same range as fp32 with fewer digits than fp16. NF4 is only how weights are *stored*. Every matmul dequantizes to bf16 first, so NF4 saves memory but doesn't speed anything up.

## Running it on your GPU {#gpu}
> Two runs on Llama 3.1 8B: an AWQ search of about 15 minutes, and an NF4 load that needs no calibration at all. Then the same perplexity eval as 3.4.

The lesson shows an **illustrative** recording first: sample data written for the course, not a real run. A healthy run looks like this:

- AWQ prints one line per block, 32 in all, with the best α (mostly 0.35 to 0.55, a few late blocks up to 0.7) and how much it cut the error compared with RTN;
- AWQ saves a 5.7 GB int4 model after about 15 minutes;
- NF4 quantizes while loading and reports 5.70 GB of weights: 3.60 GB of NF4 Linear layers plus 2.10 GB of bf16 embeddings and lm_head.

| Method | Perplexity | Change |
|---|---|---|
| fp16 | 6.24 | |
| GPTQ int4 g128 (3.4) | 6.52 | +4.5% |
| AWQ int4 g128 | 6.55 | +5.0% |
| NF4 + double quant | 6.63 | +6.3% |

You record the better of AWQ and NF4 (under 7% passes) and the NF4 model's memory (under 6.5 GB).

## Words you'll see {#glossary}
> The terms in this lesson's cards, challenges and metrics.

- **AWQ**: activation-aware weight quantization; stretches salient weight columns before quantizing.
- **Salient channel**: an input channel whose activations are large, so its weights matter most.
- **Scaling factor s**: what a weight column is multiplied by before quantizing and divided by after.
- **α (alpha)**: the power that turns average |x| into s; searched per layer. 0 means plain RTN.
- **Grid search**: trying a short list of values and keeping the best.
- **RTN**: round-to-nearest, the plain method from 3.1 and 3.2.
- **Normal distribution**: the bell curve that trained weights roughly follow.
- **Quantile**: a cut point that splits values into equal shares.
- **NF4 (4-bit NormalFloat)**: 16 levels at normal quantiles in [−1, 1], with an exact zero.
- **Codebook / palette**: a small table of allowed values; codes are positions in it.
- **Code (index)**: a weight's stored number, 0 to 15, pointing at a level.
- **Block / absmax**: the 64 weights that share one scale, and that scale.
- **argmin**: the position of the smallest value.
- **uint8**: an unsigned byte, 0 to 255. Used as an index, PyTorch treats it as a mask.
- **Double quantization**: quantizing the per-block constants themselves to 8 bits.
- **bf16 (bfloat16)**: a 16-bit float with fp32's range; the compute type for NF4.
- **bitsandbytes**: the library that provides NF4 in PyTorch.
- **QLoRA**: training LoRA adapters on top of an NF4 model; lessons 4A.1 and 4A.2.
- **lm_head**: the model's last layer, which turns hidden states into a score for every token.
- **Weight memory**: bytes of weights as loaded on the GPU.
