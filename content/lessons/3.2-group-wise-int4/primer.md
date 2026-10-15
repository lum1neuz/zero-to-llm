---
gist: At 4 bits a weight has only 15 places to land, so one big weight can blur a whole row. Giving every small group of neighbours its own scale fixes most of that for an eighth of a bit per weight.
minutes: 16
---

## What this lesson is for {#why}
> You'll take 3.1's absmax idea down from 8 bits to 4, find out why one scale per row stops working there, and fix it with one scale per group of weights.

In 3.1 you stored each weight in one byte (int8) and halved the memory of your model's Linear layers. This lesson halves it again: 4 bits per weight, called **int4**. That's how people run 7B and 8B models on ordinary gaming GPUs.

Going from 8 bits to 4 isn't just "the same thing, smaller". With 4 bits there are so few values to choose from that 3.1's recipe, one scale for a whole row, falls apart. The fix is simple and used almost everywhere: split each row into small **groups** and give every group its own scale.

By the end you'll be able to:

1. Say why int4 has 15 usable levels, not 255.
2. Show with numbers why one large weight ruins a row at 4 bits.
3. Lay a row out as groups and follow the tensor shapes through quantizing and back.
4. Work out what the extra scales cost in bits per weight, and pick a group size.

## What you need from before {#before}
> From 3.1: absmax picks a scale so the largest |w| lands on the edge of the integer range, rounds everything else to the nearest step, and the error is at most half a step.

Absmax quantization in four lines of arithmetic:

1. Find the largest magnitude in the row: **amax** = max |w|.
2. Pick the step size: **scale** = amax / 127, so amax lands exactly on ±127.
3. Quantize: **q** = round(w / scale), stored as int8.
4. Dequantize: **ŵ** = q × scale, back to floats.

`scale` is the distance between two neighbouring levels, like the spacing between marks on a ruler. Rounding to the nearest mark is never off by more than half a mark, so every weight comes back within **scale / 2** of where it started. That bound is what 3.1's round-trip test checked.

You also met 3.1's weak spot in its quiz: one huge weight makes the scale huge for everything it shares that scale with. At 8 bits you mostly got away with it. At 4 bits you won't.

## Why int4 has only 15 levels {#levels}
> 4 bits make 16 bit patterns, which as a signed number means −8 to 7. Absmax uses −7…7 so the grid is symmetric, which leaves 15 levels.

Every extra bit doubles the number of patterns, so every bit you remove halves them:

| Bits | Patterns | Signed range | Absmax uses | Levels |
|---|---|---|---|---|
| 8 | 256 | −128…127 | −127…127 | 255 |
| 4 | 16 | −8…7 | −7…7 | 15 |

Why leave −8 unused? For the same reason 3.1 left −128 unused. If you divided by 8, a positive amax would land on +8, and +8 doesn't exist in int4. Dividing by 7 keeps the grid symmetric around zero: the biggest positive and the biggest negative weight both fit.

Here's the int4 grid for a row whose amax is 2.10, so scale = 2.10 / 7 = 0.30:

```
level: -7   -3  -1 0 1   3    7
value: -2.1 -0.9 -.3 0 .3 0.9  2.1
       |....|....|.|.|....|....|
       15 marks, 0.30 apart
```

Every weight in that row must land on one of those 15 marks. Anything smaller than 0.15 in size rounds to 0.

PyTorch has no int4 type. While you work, you keep the int4 values in an int8 tensor. When the model is saved for real, two 4-bit values are packed into each byte.

## One scale per row breaks at 4 bits {#one-scale}
> With 15 levels, a single large weight sets a step so wide that every ordinary weight in its row rounds to −1, 0 or 1.

Take 8 weights, with one large one at 2.40:

```
w = [0.12, -0.31, 0.05, 0.22,
     -0.18, 0.09, 2.40, -0.27]
```

One scale for the whole row: amax = 2.40, scale = 2.40 / 7 ≈ 0.343. Divide and round:

| w | w / scale | q | back (q × scale) | error |
|---|---|---|---|---|
| 0.12 | 0.35 | 0 | 0.000 | 0.120 |
| −0.31 | −0.90 | −1 | −0.343 | 0.033 |
| 0.05 | 0.15 | 0 | 0.000 | 0.050 |
| 0.22 | 0.64 | 1 | 0.343 | 0.123 |
| −0.18 | −0.53 | −1 | −0.343 | 0.163 |
| 0.09 | 0.26 | 0 | 0.000 | 0.090 |
| 2.40 | 7.00 | 7 | 2.400 | 0.000 |
| −0.27 | −0.79 | −1 | −0.343 | 0.073 |

Seven ordinary weights squeeze onto just three values: −0.343, 0 and 0.343. Several errors are as big as the weights themselves. The 2.40 comes back perfectly, but it's the only one.

The same row in int8 has scale 2.40 / 127 ≈ 0.019, so the worst error is about 0.009. Same weights, same recipe, but the step is 18 times wider at int4. Think of measuring ants with a ruler marked only in metres: the big object fits the ruler, and everything small reads as 0 or 1.

## Groups: one scale for a few neighbours {#groups}
> Split the row into groups of g neighbours and give each group its own absmax scale. A large weight then only coarsens its own group.

Split the same 8 weights into two groups of 4:

| | Group A | Group B |
|---|---|---|
| Weights | 0.12, −0.31, 0.05, 0.22 | −0.18, 0.09, 2.40, −0.27 |
| amax | 0.31 | 2.40 |
| scale = amax / 7 | 0.0443 | 0.343 |
| q | 3, −7, 1, 5 | −1, 0, 7, −1 |
| errors | 0.013, 0, 0.006, 0.001 | 0.163, 0.090, 0, 0.073 |

Group A no longer shares a scale with the 2.40. Its step shrank from 0.343 to 0.044, and its errors dropped to almost nothing. Group B still contains the 2.40, so it's exactly as blurry as before.

That's the whole idea, and also its limit: **groups contain an outlier's damage, they don't remove it.** The mean error over the 8 weights halves (0.081 → 0.043), but the worst error doesn't change, because it lives in the outlier's own group. Lesson 3.3 deals with the outlier itself.

You store one scale per group instead of one per row. That's the price, and a later section works out what it costs.

## Groups as a reshape {#shapes}
> Fold the row into a table with one group per row, take each row's absmax as a column of scales, and broadcasting divides every group by its own scale.

The neat trick is that you don't need a loop over groups. A row of 16 weights in groups of 4 becomes a 4 × 4 table, one group per row:

```
w, shape [16]
0.12 -0.31 0.05 0.22 -0.18 ...

grouped, shape [4, 4]
[ 0.12 -0.31  0.05  0.22]  group 0
[-0.18  0.09  2.40 -0.27]  group 1
[ 0.15 -0.08  0.30 -0.11]  group 2
[-0.25  0.19 -0.04  0.28]  group 3
```

`reshape` folds a tensor into a new shape without changing the order of the numbers. You can write `-1` for one of the sizes, meaning "work this one out from the total". For example, with 12 numbers:

```python
x = torch.arange(12)
x.reshape(-1, 4).shape
# torch.Size([3, 4])
```

Then the shapes through the rest of the round trip:

| Step | Shape | What it is |
|---|---|---|
| grouped weights | `[4, 4]` | one group per row |
| one scale per group | `[4, 1]` | a column, one per row |
| divide, round | `[4, 4]` | ints from −7 to 7 |
| multiply back | `[4, 4]` | floats again |
| flatten | `[16]` | lines up with `w` again |

The scales need to be a `[4, 1]` column so that broadcasting stretches each row's scale across that row, exactly like dividing counts by row totals in 0.0. To get the largest magnitude of each row, use `amax` along `dim=1` and keep the dimension:

```python
t = torch.tensor([[0.5, -2.0],
                  [0.3,  0.1]])

t.abs().amax(dim=1, keepdim=True)
# [[2.0], [0.3]]  shape [2, 1]

t.abs().amax(dim=1)
# [2.0, 0.3]      shape [2]
```

`amax` returns only the values. `max(dim=1)` also works, but it returns a pair, `(values, indices)`, and you have to take `.values` from it.

## Mistakes the tests look for {#mistakes}
> Most wrong answers here are shape mistakes. A few divide by the wrong limit. The tests name each one.

| Mistake | What goes wrong |
|---|---|
| No `keepdim=True` | Scales have shape `[4]` and line up with the **columns**. On a square table there's no error, just wrong numbers. |
| `reshape(g, -1)` | Makes g rows instead of rows of g, so a "group" isn't g neighbours. |
| `amax(dim=0)` | One scale per column, shared across groups. |
| `max(dim=1) / 7` | `max` with a dim returns a pair, not a tensor, so the division fails. |
| Divide by 127 | int8's limit from 3.1. The biggest weight lands on 127, far outside int4. |
| Divide by 8 | +8 doesn't fit in int4. |
| Divide by 15 | 15 is the top of *unsigned* 4-bit (0…15). Signed int4 stops at 7. |
| No flatten at the end | The result is `[4, 4]`, but `w` is `[16]`. |
| Flatten before multiplying | `[16]` times `[4, 1]` broadcasts to `[4, 16]`: every scale hits the whole row. |

The first one deserves a second look. It's the same silent bug as makemore's square table in 0.0: if the number of groups equals the group size, the wrong broadcast still runs. That's why the tests also try 24 weights in groups of 8, where the table is `[3, 8]` and the bug can't hide.

## What a scale costs {#overhead}
> Each group stores one 16-bit scale, so every weight pays 16 / g extra bits on top of its own 4. With groups of 128 that's 4.125 bits per weight.

Scales are kept as fp16, 16 bits each. A group of g weights shares one, so each weight carries 16 / g bits of scale:

| Group size g | Scale bits per weight | Total bits per weight |
|---|---|---|
| 4 | 16 / 4 = 4 | 8 |
| 32 | 0.5 | 4.5 |
| 64 | 0.25 | 4.25 |
| 128 | 0.125 | 4.125 |
| a 2048 row | ≈ 0.008 | ≈ 4.008 |

Look at the first row: groups of 4 cost 8 bits per weight, as much as int8. The lesson uses g = 4 on 16 weights only because it's small enough to read. Real models use groups of 32 to 128, where the overhead is small.

For Llama 3.1 8B with g = 128: 8 × 10⁹ weights × 4.125 bits ≈ 4.1 GB, against 4.0 GB if scales were free and 16 GB in fp16.

## Picking a group size {#group-size}
> Smaller groups mean lower error and more scale bits. 128 is the common default, and g must divide the row length.

On the lesson's 16-weight row, mean error at int4 for different group sizes:

| g | Groups | Mean error | Bits per weight |
|---|---|---|---|
| 16 (whole row) | 1 | 0.086 | 5 |
| 8 | 2 | 0.047 | 6 |
| 4 | 4 | 0.026 | 8 |

```chart
kind: bar
title: Mean int4 error against group size
x: { label: group size g, values: [16 (row), "8", "4"] }
y: { label: mean error, min: 0, max: 0.1 }
series:
  - { name: mean error, values: [0.086, 0.047, 0.026] }
caption: On the lesson's 16-weight row, each halving of the group size cuts the mean error roughly in half.
```

(The bits column is so high only because the row is tiny. In a real 2048-wide row these become 4.008, 4.06 and 4.125 bits for g = 2048, 256 and 128.)

The pattern holds in real models: halve the group size and the error drops, but the scale overhead doubles. Things to weigh:

- **Common sizes.** llama.cpp's Q4_0 format uses blocks of 32. Most GPTQ and AWQ checkpoints use 128.
- **It has to divide the row.** A group can't straddle two rows. Every Linear row in your 30M model has 512 or 2048 weights, and both divide by 128.
- **Diminishing returns.** Going from per-row to 128 fixes most of the damage. Going from 128 to 32 helps less and costs four times as many scales.

## What it means for your model {#memory}
> Only the 24 Linear layers are quantized: 37.7 MB in fp16 becomes about 9.7 MB at int4 with groups of 128, and the whole model goes from 60.4 to about 32.4 MB.

Each of the 6 blocks has four Linear layers. Together they hold 18,874,368 weights. Embeddings and LayerNorm stay in fp16, as in 3.1.

| | Linear weights | Whole model |
|---|---|---|
| fp16 | 37.7 MB | 60.4 MB |
| int8 per row (3.1) | 18.9 MB | 41.6 MB |
| int4, g = 128 | 9.7 MB | 32.4 MB |

The 9.7 MB is 18,874,368 weights × half a byte = 9.44 MB, plus 147,456 group scales × 2 bytes = 0.29 MB. That 9.7 MB is the systems metric you record, and the pass limit is 10.0 MB.

## The replay and your own run {#replay}
> You'll run one script that quantizes every Linear with groups of 128 and evaluates perplexity. It takes a few minutes, and healthy output shows small per-layer errors, one flagged layer and a few percent of perplexity cost.

The run uses the course repo's `quantize_model.py` with `--scheme absmax-int4 --group-size 128` on your TinyStories 30M checkpoint. Most of the few minutes go to the perplexity evaluation on 2,000 samples.

The replay in the lesson is an **illustrative recording**: sample data written for the course, not a real run. Use it to learn what to look for, then compare with your own numbers:

- **A per-row int4 baseline first.** Perplexity gets much worse, about +27% in the recording. That's the problem this lesson fixes.
- **One line per Linear layer** with its max error, mostly around 0.01 to 0.02.
- **One red layer**, `blocks.4.mlp.c_proj`, much higher than the rest. That's the same outlier as in 3.1's replay. It now only blurs its own group of 128, not its whole row.
- **Memory and perplexity at the end.** Linears 37.7 → 9.7 MB, perplexity 7.84 → 8.12, a cost of about +3.6%. The pass limit is +5%.

If your perplexity cost is far above 5%, check that you grouped along each row and that every group got its own scale.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards and challenges.

- **int4**: a 4-bit integer. Signed, it holds −8…7. Absmax uses −7…7.
- **Level**: one of the values a quantized weight can land on. int4 absmax has 15.
- **Absmax**: picking the scale from the largest magnitude, so it lands exactly on the edge of the range.
- **Scale**: the step between neighbouring levels. One number per row in 3.1, one per group here.
- **Group (block)**: g neighbouring weights in a row that share one scale.
- **Group size (g)**: how many weights share a scale. 128 is common.
- **Per-row quantization**: one scale per row. The same as a group size equal to the row length.
- **Outlier**: a weight far larger than its neighbours. It sets the scale for everything that shares it.
- **Bits per weight**: the weight's own bits plus its share of the scale, 4 + 16 / g.
- **Packing**: storing two 4-bit values in one byte.
- **amax**: the largest value along a dimension, without the indices that `max(dim=…)` also returns.
- **reshape**: fold a tensor into a new shape with the same numbers in the same order. `-1` means "work it out".
- **keepdim**: keep a reduced dimension as size 1, so the result still broadcasts row by row.
