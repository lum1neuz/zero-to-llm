---
paper: "The case for 4-bit precision: k-bit Inference Scaling Laws"
gist: For a fixed budget of model bits, a larger model stored in 4 bits almost always beats a smaller one in 8 or 16 bits, and small quantization blocks are part of what makes 4 bits work.
---

## The question {#question}
> If memory is the limit, should you spend your bits on more parameters or on more precision per parameter?

Inference memory is roughly parameters × bits per parameter. You could run a small model in fp16 or a model four times larger in 4 bits for about the same memory. The paper asks which choice gives better accuracy, and whether the answer holds across model families and sizes. It counts every bit, including the extra bits that scales cost.

## How they measured it {#setup}
> A large sweep of model families, sizes and bit widths, scored by zero-shot accuracy against the total bits the model occupies.

They quantize pretrained models from several open families (among them OPT, BLOOM and Pythia), from tens of millions of parameters up to BLOOM's 176B. Each is quantized at bit widths from 3 to 8 and compared with 16-bit, and evaluated zero-shot on standard tasks without any fine-tuning. Plotting accuracy against total model bits turns each setting into a scaling curve, so two precisions can be compared at equal memory.

## 4 bits is the sweet spot {#four-bits}
> At equal total bits, 4-bit models are almost always the most accurate, and going lower stops paying off.

Moving from 16 to 8 to 4 bits frees memory faster than it costs accuracy, so the freed bits are better spent on a bigger model. At 3 bits that stops holding reliably: the quality loss per parameter grows faster than the extra parameters can make up for. That's why 4-bit weights became the default target for running large models on consumer GPUs.

## Small blocks help {#blocks}
> Quantizing in small blocks, each with its own scale, improves results at low precision, even after paying for the extra scale bits.

A block here is what this lesson calls a group: a run of consecutive weights sharing one scale. A smaller block means an outlier only distorts its own block, so the rest of the weights keep a finer grid. Each block's scale costs extra bits per weight (16 / block size for a 16-bit scale), and the paper includes that cost in its bit counts. Small blocks still come out ahead, which is the case for group-wise int4.

## Data types matter too {#data-types}
> At 4 bits, how you place the 16 levels makes a difference: float-like and quantile-based codes beat evenly spaced integers.

An integer grid spaces its levels evenly, but trained weights cluster near zero with a few large values. Codes that put more levels near zero, such as small floating-point formats or quantile quantization, waste fewer levels and lose less accuracy. This idea leads to NF4, which lesson 3.5 covers.

## What it doesn't settle {#limits}
> The study covers zero-shot accuracy with simple round-to-nearest quantization; smarter methods can shift the picture below 4 bits.

All models are quantized directly, without calibration or error correction. Methods that correct rounding error using data, like GPTQ in lesson 3.4, could make 3 bits more competitive. Zero-shot accuracy is also only one way to judge a model.

## How the course uses it {#course}
> 3.2 builds the block idea by hand, int4 with one absmax scale per group, and counts scale bits the same way the paper does.

Your calc card's 4 + 16 / 128 = 4.125 bits per weight is the paper's accounting. The g = 16 versus g = 4 comparison in the Build step is a tiny version of its block-size result: smaller groups, lower error, more bits spent on scales.
