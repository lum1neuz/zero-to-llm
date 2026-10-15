---
gist: Quantization stores each weight as a small whole number plus one shared scale, so a model takes half the memory or less; this lesson builds the simplest version, int8 absmax, by hand.
minutes: 15
---

## What this lesson is for {#why}
> Big models don't fit on your GPU in their normal 16-bit form. Storing weights in fewer bits fixes that, and this lesson builds the simplest way to do it.

Llama 3.1 8B has about 8 billion weights. In its normal format each weight takes 2 bytes, so the weights alone need 16 GB. A typical gaming GPU has 12 GB of memory. The model doesn't fit before you've run a single token.

**Quantization** means storing each weight in fewer bits. Store them in 1 byte each and the same model needs about 8 GB, which fits. Store them in half a byte and it's about 4 GB.

Phase 3 builds this up step by step:

| Lesson | What it adds |
|---|---|
| 3.1 (this one) | int8: one byte per weight, one scale per row |
| 3.2 | int4: half a byte, one scale per small group |
| 3.3 | outliers: the few huge values that break it |
| 3.4 | GPTQ: use real data to round smarter |
| 3.5 | AWQ and NF4: two more ways to spend 4 bits well |
| 3.6 | llama.cpp: the formats people actually ship |

It matters for the rest of the course too. The Mongolian fine-tune in phase 4 trains on top of a 4-bit model, and phase 5 serves quantized models.

Here you write the core by hand on one row of 8 numbers, then apply it to every weight matrix in the 30M TinyStories model you trained in phase 1, and measure what it cost.

## What you need from before {#before}
> A layer's weights are a `[out, in]` grid of floats, perplexity measures how surprised a model is, and decoding reads every weight for every token.

Three ideas from phases 0–2 come back here.

**Linear layers.** `nn.Linear(512, 2048)` holds a weight tensor of shape `[2048, 512]`: one **row** per output, one column per input. Most of a transformer's weights live in these layers. Your 30M model has 24 of them (4 in each of its 6 blocks). The rest is mainly the token embeddings.

**Perplexity.** The exponential of the average loss on held-out text. Roughly, how many next tokens the model is hesitating between. Lower is better. If quantization damages the model, perplexity goes up, so it's the quality measure in every phase 3 lesson.

**Decoding reads all the weights.** In phase 2 you saw that generating each token reads every weight from GPU memory once. Smaller weights mean less to read, so quantization can also make generation faster, not just smaller.

From phase 0 you also need `abs()`, `amax(dim=…, keepdim=True)` and broadcasting. They come back in 3.2.

## Bytes per weight {#bytes}
> Memory is parameters × bytes per parameter. Halve the bytes and you halve the memory.

This is the arithmetic behind the whole phase. 1 GB here means 10⁹ bytes.

| Format | Bytes per weight | Llama 3.1 8B (8.0 × 10⁹) |
|---|---|---|
| fp32 | 4 | 32 GB |
| fp16 | 2 | 16 GB |
| int8 | 1 | 8 GB |
| int4 | 0.5 | 4 GB |

Your 30M model is small enough to check by hand. Its 24 Linear layers hold about 18.9 million weights:

```
fp16:  18.9 M × 2 bytes = 37.7 MB
int8:  18.9 M × 1 byte  = 18.9 MB
```

The embeddings and LayerNorms stay in fp16 in this lesson, so the whole file goes from 60.4 MB to about 41.6 MB, not to half. Real quantized models usually make the same choice.

You may see GiB elsewhere (2³⁰ bytes, about 7% bigger than a GB). Tools mix them up, so check which one a number uses.

## What a number format is {#formats}
> fp16 is a float with about 3 significant digits; int8 is a whole number from −128 to 127, exactly Java's `byte`.

A **float** stores a number like scientific notation: a sign, a few significant digits and an exponent. **fp16** ("half precision") uses 16 bits. It keeps about 3 decimal digits and reaches up to 65,504. Its values sit close together near zero and far apart for big numbers. Java's `float` is fp32, the 32-bit version.

An **integer** format stores whole numbers, evenly spaced. **int8** uses 8 bits, so it has 2⁸ = 256 values: −128 to 127. That's exactly Java's `byte`, including what happens when you go past the end:

```java
// Java
byte b = (byte) 128;  // -128
byte c = (byte) 200;  // -56
```

Values wrap around instead of stopping at 127. PyTorch's int8 does the same, and it's behind one of the mistakes the tests check for.

| Format | Bits | Distinct values | Holds |
|---|---|---|---|
| fp16 | 16 | about 63,000 | decimals, wide range |
| int8 | 8 | 256 | −128 … 127 |
| int4 | 4 | 16 | −8 … 7 |

Weights are small decimals like 0.42 or −1.30. int8 can't store 0.42 at all. The trick is a **scale**.

## The idea: a ruler with 255 ticks {#scale}
> Pick a step size, the scale. Store each weight as how many steps it is from zero, a whole number, and keep the scale once for the whole row.

Think of a ruler with evenly spaced ticks. You record each weight as the tick nearest to it, a whole number, and you keep one number that says how far apart the ticks are. To read a weight back, multiply its tick number by the spacing.

```
q     = round(w / scale)
w_hat = q × scale
```

`q` is the stored integer and `w_hat` ("w hat") is the value you get back. They're close, but not equal. The difference is the **quantization error**.

It's like saving a photo as a JPEG. You keep something that looks almost the same in far fewer bytes, and you accept a small, controlled loss.

The cost of the scale is tiny. One fp16 scale (2 bytes) shared by a 512-weight row adds 0.4%.

## Absmax, worked by hand {#absmax}
> Absmax picks the scale so the weight with the biggest magnitude lands exactly on ±127.

The ruler should be just long enough to reach the biggest weight in the row, whether it's positive or negative. So:

```
scale = max(|w|) / 127
```

Take a row of four weights: `[0.81, −2.54, 0.13, 1.50]`. The biggest magnitude is 2.54, so the scale is 2.54 / 127 = 0.02.

| w | w / scale | q (rounded) | q × scale | error |
|---|---|---|---|---|
| 0.81 | 40.5 | 40 | 0.80 | 0.01 |
| −2.54 | −127.0 | −127 | −2.54 | 0 |
| 0.13 | 6.5 | 6 | 0.12 | 0.01 |
| 1.50 | 75.0 | 75 | 1.50 | 0 |

You store `[40, −127, 6, 75]` as four bytes plus the scale 0.02. The biggest weight comes back exactly, and nothing is ever out of range.

**Why 127 and not 128?** int8 goes from −128 to 127. Divide by 128 and a positive biggest weight lands on +128, which doesn't exist. It wraps to −128, like Java's `(byte) 128`, and the largest weight flips sign. Dividing by 127 keeps the grid symmetric and leaves −128 unused.

**Why the magnitude?** If the scale came from the largest *value*, a row like `[1.0, −3.0]` would get its scale from 1.0, and −3.0 would land on −381, far outside int8. The weight that matters is the one biggest in size, whichever sign it has.

## Rounding and the error bound {#rounding}
> Rounding to the nearest tick means no weight is ever more than half a step off: error ≤ scale / 2.

Each weight sits somewhere between two ticks. Rounding picks the nearer one, so it moves by at most half the gap. In the table above that's 0.01, exactly half of 0.02. The tests check this bound.

Look at 40.5 and 6.5. They're exactly halfway. `torch.round` sends halves to the **nearest even** number, so 40.5 → 40 and 6.5 → 6. That's called round half to even. It avoids a tiny bias upwards, and it explains some answers that look off by one.

Two other ways to turn decimals into integers break the bound:

| Method | 6.7 becomes | Max error |
|---|---|---|
| Round to nearest | 7 | half a step |
| Truncate (cast) | 6 | almost a whole step |
| Floor | 6 | almost a whole step |

Casting a float straight to int8 **truncates**, like Java's `(int) 6.7`, which is 6. That's a classic mistake: round first, then convert.

## Why one scale per row {#per-row}
> Each row gets its own scale, so one unusual row can't ruin the others.

A weight matrix `W` has shape `[out, in]`. You could use one scale for the whole matrix, but then the single biggest weight anywhere sets the step size for millions of others. Instead each row gets its own:

```
W         [2048, 512]
scales    [2048, 1]
codes q   [2048, 512]  (int8)
```

Within a row the problem doesn't go away, though. Suppose one weight is 40.0 and the rest are under 1.0. The scale is 40 / 127 ≈ 0.31, so every weight smaller than about 0.16 rounds to 0. One **outlier** has flattened its whole row. That's the main problem of lesson 3.3, and you'll see one such row in this lesson's recording.

In the challenge you quantize a single row of 8 weights, so there's one scale.

## Fewer bits, bigger steps {#bits}
> Every bit you remove halves the number of levels and doubles the step, so the error roughly doubles too.

With b bits, symmetric absmax uses levels from −(2ᵇ⁻¹ − 1) to +(2ᵇ⁻¹ − 1). For a row whose biggest weight is 2.10:

| Bits | Top level | Step (2.10 / top) | Worst error |
|---|---|---|---|
| 8 | 127 | 0.0165 | 0.008 |
| 6 | 31 | 0.0677 | 0.034 |
| 4 | 7 | 0.3000 | 0.150 |

```chart
kind: bar
title: Worst rounding error against bit width
x: { label: bits per weight, values: [8 bits, 6 bits, 4 bits] }
y: { label: worst error, min: 0, max: 0.16 }
series:
  - { name: worst error, values: [0.008, 0.034, 0.150] }
highlight: { x: 4 bits, label: "0.150" }
caption: For a row whose biggest weight is 2.10, every two bits you remove make the worst error about four times bigger.
```

The slider card lets you drag the bit width and watch the error. At 4 bits a step of 0.3 is bigger than many weights themselves. That's why 3.2 has to do something smarter than one scale per row.

## The way back: dequantize {#dequantize}
> To use the weights you turn them back into floats: integer times scale. Convert to float first, or the arithmetic stays in int8 and overflows.

This lesson does **weight-only** quantization. The weights are stored in int8, and right before a layer runs they're turned back into floats (**dequantized**) and the matmul runs in fp16 as usual. You save memory. You don't change the math.

Going back undoes going forward. Quantizing *divided* by the scale, so going back *multiplies*. Dividing again is a common slip.

Convert the integers to floats before multiplying. In int8, 127 × 2 is 254, which wraps to −2, the same wrap-around as Java's `byte`. Floats don't have that problem.

The dequantize function should use the `q` and `scale` it receives as arguments, not variables lying around in the script. The tests call it on other rows to check that.

## Running it on your GPU {#gpu}
> The run quantizes all 24 Linear layers of your 30M model, prints the max error of each, and compares perplexity and memory with fp16.

`quantize_model.py` does what you built, for every row of every Linear layer, then runs the same perplexity eval as your phase 1 baseline. It takes a few minutes. The lesson shows a recording first. Like every replay in this phase, it's **illustrative** sample data, labelled as such, not a real run.

A healthy run looks like this:

- one line per layer with its **max error**, mostly around 0.003 to 0.006;
- one layer far above the rest, `blocks.4.mlp.c_proj` at about 0.04, an outlier row;
- Linear weights 37.7 → 18.9 MB, the whole model 60.4 → 41.6 MB;
- perplexity 7.84 → 7.91, a cost of +0.9%;
- a sample story that still reads normally.

You record two numbers. The perplexity change should stay under 2%, and the int8 Linear weights (plus their scales) under 19.5 MB. If perplexity jumps by much more, look for the layer whose max error stands out.

## Words you'll see {#glossary}
> The terms in this lesson's cards, challenges and metrics.

- **Quantization**: storing numbers with fewer bits, as whole numbers plus a scale.
- **fp32 / fp16**: 32-bit and 16-bit floating point. fp16 is the normal format for model weights.
- **int8 / int4**: whole numbers in 8 bits (−128…127) or 4 bits (−8…7).
- **Scale**: the step size between neighbouring levels. One per row here.
- **Absmax**: choosing the scale as the largest magnitude divided by the top level (127 for int8).
- **amax**: the largest absolute value, max(|w|).
- **q**: the stored integer codes.
- **Dequantize**: turning codes back into floats, q × scale. The result is often written **w_hat**.
- **Round trip**: quantize, then dequantize, then compare with the original.
- **Quantization error**: the difference between a weight and its round-trip value. Never more than scale / 2 with rounding.
- **Round half to even**: `torch.round`'s rule for exact halves (6.5 → 6, 7.5 → 8).
- **Clamp**: limit values to a range, here −127…127.
- **Overflow / wrap-around**: going past an integer type's range and coming out the other end, like (byte) 128 = −128.
- **Per-row (per-channel) scale**: one scale for each output row of a weight matrix.
- **Outlier**: one value far bigger than its neighbours, which inflates the shared scale.
- **Weight-only quantization**: storing weights in int8 but running the matmul in floats.
- **Perplexity (ppl)** and **ppl delta**: the quality measure, and its change in % from fp16.
- **VRAM**: the GPU's own memory, where the weights have to fit.
