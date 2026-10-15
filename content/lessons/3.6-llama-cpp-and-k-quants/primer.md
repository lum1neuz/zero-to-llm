---
gist: llama.cpp's k-quants give every 32 weights a scale and a starting value, then store those numbers in 6 bits inside a 256-weight super-block; that's how a 16 GB model becomes a 4.9 GB file that runs fast on a 12 GB card.
minutes: 14
---

## What this lesson is for {#why}
> llama.cpp runs quantized models on almost any hardware from one file. Its k-quant formats combine ideas from this whole phase, and you'll build one by hand.

**llama.cpp** is a C/C++ program that runs language models on GPUs, CPUs and laptops. It reads models from a single **GGUF** file: weights, tokenizer and settings in one place. Quantized GGUF files are how most people run open models on their own machines, and in phase 5 you'll serve one with llama.cpp's server and race it against vLLM.

Its main family of formats is the **k-quants**, with names like Q4_K_M. They stack two ideas you already know:

- 3.2's groups: every 32 weights get their own scale, plus a starting value so the grid fits the group's actual range;
- 3.5's double quantization: those scales and starting values are themselves quantized, inside a bigger **super-block** of 256 weights.

You'll build one Q4_K super-block, check its error and work out its bits per weight. Then you'll watch Llama 3.1 8B go through llama.cpp's own tools: convert, quantize, measure perplexity, measure speed.

## What you need from before {#before}
> Group-wise absmax, double quantization, perplexity, and the fact that decoding reads every weight once per token.

**Groups (3.2).** Each small block of weights gets its own scale, so one large weight only affects its neighbours.

**Double quantization (3.5).** The per-block numbers cost bits too, so you quantize them against one higher-precision number.

**Bits per weight (3.2, 3.5).** A format's real cost is the weight's own bits plus its share of every scale. 4 bits with one fp16 scale per 128 weights is 4 + 16 / 128 = 4.125.

**Decode is memory-bound (phase 2).** Generating one token reads every weight once. The fewer bytes the weights take, the more tokens per second.

## Symmetric grids waste space {#asym}
> A grid centred on zero wastes levels when a group's values are lopsided. Starting the grid at the group's lowest value fits it to the real range.

3.2's absmax puts its grid symmetrically around zero, from −max|w| to +max|w|. Take a group whose weights run from lo = −0.06 to hi = 0.24. The grid has to reach −0.24 too, though nothing is down there.

Q4_K uses a different grid. Code 0 means lo, code 15 means hi, and the 16 codes are spread evenly between:

```
step = (hi − lo) / 15
q    = round((w − lo) / step)
back = q × step + lo
```

Why 15 and not 16? Like fence posts: 16 posts have 15 gaps between them. Divide by 16 and the steps come out too small, so hi needs code 16, which doesn't exist.

For the group above:

| | Symmetric int4 | Scale + min |
|---|---|---|
| Range covered | −0.24 … 0.24 | −0.06 … 0.24 |
| Levels | 15 | 16 |
| Step | 0.24 / 7 ≈ 0.034 | 0.30 / 15 = 0.02 |
| Worst error | 0.017 | 0.01 |

Check a weight, w = 0.075. With scale + min: (0.075 + 0.06) / 0.02 = 6.75, which rounds to 7, and 7 × 0.02 − 0.06 = 0.08, off by 0.005. Symmetric: 0.075 / 0.034 ≈ 2.19 → 2 → 0.069, off by 0.006.

Codes count up from lo, so they're never negative and fit in an unsigned byte, `uint8` (0 to 255). That's a trap if you use a symmetric formula by mistake: a negative code like −3 doesn't fit and wraps to 253. The tests spot codes above 15 and tell you why.

Other mistakes the tests look for: counting steps from zero instead of from lo (every negative weight lands on code 0), truncating instead of rounding, and a step that only covers 0…hi.

On the lesson's 256 weights, scale + min cuts the mean error from about 0.0040 to 0.0031, about 20%, with the same 4-bit codes.

## The price of a scale and a min {#cost}
> Two fp16 numbers per 32 weights cost a whole extra bit per weight.

Every group of 32 now carries two numbers, its step and its lo. In fp16 that's 32 bits per 32 weights:

```
4 + (16 + 16) / 32 = 5 bits
```

A 25% overhead on a 4-bit format. That's what the super-block fixes.

## Super-blocks: quantize the scales {#superblock}
> Eight groups of 32 form a super-block of 256. Their 8 steps and 8 mins are stored as 6-bit integers against two fp16 numbers for the whole super-block.

A super-block holds 8 groups of 32 weights, called **sub-blocks**. Their 8 steps are all small positive numbers, so you quantize them with absmax, exactly like 3.1, but onto a 6-bit range:

- 6 bits hold 0 to 63;
- one fp16 number **d** per super-block is the step for the steps, so the largest one lands on 63;
- each step is stored as a whole number of d's.

The 8 mins get the same treatment with their own fp16 number, **dmin**. llama.cpp stores the mins as positive numbers (minus lo), so both lists are 0 to 63.

A small example with three steps, `[0.0110, 0.0150, 0.0200]`:

```
d = 0.0200 / 63 ≈ 0.000317

0.0110 / d = 34.65 → 35
0.0150 / d = 47.25 → 47
0.0200 / d = 63.00 → 63

back: 35 × d ≈ 0.01111
      47 × d ≈ 0.01492
```

The steps come back within a fraction of a percent. Decoding a weight then uses both levels:

```
w ≈ d · sc · q − dmin · m
```

where q is the weight's 4-bit code, sc and m are its sub-block's 6-bit step and min, and d and dmin are the super-block's fp16 numbers.

The tests check the 6-bit absmax: one d for all the values (not one per value), the largest landing on 63 (not 64, which doesn't fit, or 31, which is only 5 bits), d the right way up, and rounding rather than truncating.

On the lesson's super-block, the 6-bit scales add only about 3% to the error, and the cost drops from 5.0 to 4.5 bits per weight.

## Counting bits per weight {#bpw}
> Add up every bit in a super-block, then divide by 256.

Here's the counting for **Q6_K**, a 6-bit k-quant whose 16 sub-blocks of 16 weights each have an 8-bit scale and no min:

| Part | Bits |
|---|---|
| 256 codes × 6 bits | 1,536 |
| 16 scales × 8 bits | 128 |
| one fp16 super-scale | 16 |
| **Total** | **1,680** |

1,680 / 256 = 6.5625 bits per weight. The card has you do the same for Q4_K: 256 four-bit codes, 8 six-bit steps, 8 six-bit mins and two fp16 numbers.

**Q8_0** is the older, simpler format: blocks of 32 int8 codes with one fp16 scale, so (32 × 8 + 16) / 32 = 8.5 bits.

## The family, and _S/_M/_L {#family}
> Each k-quant type has a fixed bits-per-weight. The _S, _M and _L variants mix types across tensors, using more bits where errors hurt most.

| Type | Bits per weight | Sub-blocks | Min? |
|---|---|---|---|
| Q8_0 | 8.5 | 32 | no |
| Q6_K | 6.5625 | 16 | no |
| Q5_K | 5.5 | 32 | yes |
| Q4_K | 4.5 | 32 | yes |
| Q3_K | 3.4375 | 16 | no |
| Q2_K | 2.625 | 16 | yes |

A file called **Q4_K_M** isn't all Q4_K. The "M" (medium) mix stores half of the attention value (`attn_v`) and feed-forward down (`ffn_down`) tensors, and the output layer, in Q6_K, because errors there hurt quality most. Everything else is Q4_K. Averaged over the whole model, that's 4.89 bits per weight.

llama.cpp can also use an **importance matrix**, made by running calibration text through the model. It tells the quantizer which weights the activations lean on, so their rounding counts more when it searches for each sub-block's step and min. That's AWQ's idea again.

## File size and speed {#size-speed}
> File size is parameters × bits per weight / 8. Decode speed is capped by memory bandwidth divided by the bytes read per token.

**Size.** Llama 3.1 8B has 8.03 × 10⁹ parameters. Q5_K_M averages 5.70 bits:

```
8.03e9 × 5.70 / 8
≈ 5.72 × 10⁹ bytes = 5.72 GB
```

The calc card asks the same for Q4_K_M.

**Decode speed.** Each new token reads every weight once. An RTX 3060 moves about 360 GB/s, so with the 8.54 GB Q8_0 file:

```
360 / 8.54 ≈ 42 tokens/s at most
```

Real runs reach most of that bound; the recording measures about 38 for Q8_0. Smaller files decode faster.

**Prompt processing** is different. The prompt's tokens are processed together, so one read of the weights serves hundreds of tokens and the GPU's arithmetic becomes the limit. llama-bench reports both: `pp512` (prompt, 512 tokens) and `tg128` (generating 128 tokens). Your metric is tg128.

## Running it on your GPU {#gpu}
> Five llama.cpp tools: convert to GGUF, build an importance matrix, quantize, measure perplexity, measure speed.

The steps, in order:

1. `convert_hf_to_gguf.py` turns the Hugging Face checkpoint into a 16 GB F16 GGUF file.
2. `llama-imatrix` runs calibration text and writes the importance matrix.
3. `llama-quantize` writes a file per type, Q8_0 down to Q2_K.
4. `llama-perplexity` measures each file on wikitext-2 at a context of 512.
5. `llama-bench` measures prompt and decode speed.

The F16 file doesn't fit in 12 GB, so for it you put only 24 of the 32 layers on the GPU with `-ngl 24`. That changes speed, not perplexity. The whole run takes about 3 hours, mostly perplexity.

The lesson shows an **illustrative** recording first: sample data written for the course, not a real run. Healthy results look like this:

| Type | File | Perplexity | Change |
|---|---|---|---|
| F16 | 16.07 GB | 6.233 | |
| Q8_0 | 8.54 GB | 6.236 | +0.05% |
| Q6_K | 6.60 GB | 6.253 | +0.3% |
| Q5_K_M | 5.73 GB | 6.297 | +1.0% |
| Q4_K_M | 4.92 GB | 6.407 | +2.8% |
| Q3_K_M | 4.02 GB | 6.889 | +10.5% |
| Q2_K | 3.18 GB | 9.751 | +56.4% |

Down to Q4_K_M the cost stays under 3%. Below 4 bits it climbs fast. Q4_K_M decodes at about 57 tokens/s on an RTX 3060; the F16 file, partly offloaded, manages about 8.

Compare perplexities only from the same tool and context length. llama-perplexity's numbers differ from the Python eval in 3.4 and 3.5, so compare F16 with Q4_K_M here, not with those lessons.

You record Q4_K_M's perplexity change (under 4% passes) and its tg128 decode speed (at least 30 tokens/s). Keep the file: lesson 5.1 serves it.

## Words you'll see {#glossary}
> The terms in this lesson's cards, challenges and metrics.

- **llama.cpp**: a C/C++ program for running language models on GPUs, CPUs and laptops.
- **GGUF**: llama.cpp's single-file model format, with weights, tokenizer and settings.
- **k-quants**: llama.cpp's family of super-block formats, Q2_K to Q6_K.
- **Q8_0**: the older 8-bit format, one fp16 scale per 32 weights.
- **Super-block**: 256 weights that share fp16 numbers d and dmin.
- **Sub-block**: a group of 16 or 32 weights inside a super-block, with its own step (and min).
- **lo / hi**: a sub-block's smallest and largest weight.
- **Scale + min (asymmetric) quantization**: code 0 means lo and code 15 means hi.
- **Symmetric quantization**: a grid centred on zero, as in 3.1 and 3.2.
- **d, dmin**: the super-block's fp16 numbers for decoding the 6-bit steps and mins.
- **sc, m**: a sub-block's 6-bit step and min.
- **6-bit**: whole numbers from 0 to 63.
- **uint8**: an unsigned byte, 0 to 255; negative values wrap around.
- **Bits per weight (bpw)**: total bits of a format divided by the number of weights.
- **Q4_K_M, _S, _L**: mixes of k-quant types across tensors; M uses Q6_K for the most sensitive ones.
- **Importance matrix (imatrix)**: activation statistics that tell llama-quantize which weights matter most.
- **F16**: the unquantized 16-bit GGUF, the baseline.
- **-ngl**: how many layers llama.cpp puts on the GPU; 99 means all of them.
- **Memory bandwidth**: how many bytes per second the GPU can read from its memory.
- **pp512 / tg128**: llama-bench's prompt-processing and token-generation (decode) tests.
- **Tokens per second (tok/s)**: decode speed, this lesson's systems metric.
- **wikitext-2**: the standard text llama-perplexity is measured on.
