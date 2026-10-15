---
paper: GPTQ
gist: Quantize each layer one column at a time and spread every column's rounding error over the columns still to come, using the inputs' Hessian, which brings 175B-parameter models to 3–4 bits in a few GPU hours.
---

## The problem {#problem}
> Round-to-nearest is fast but loses too much at 3–4 bits; earlier accurate methods were far too slow for models with billions of parameters.

Post-training quantization with no retraining was well understood for small models. The accurate methods solved a per-layer reconstruction problem but scaled badly with layer size. Simple rounding scaled fine but broke down at low bit widths on large models. GPTQ aims for both: accuracy close to the expensive methods at a cost that works on GPT-scale models.

## The layer-wise objective {#objective}
> Choose quantized weights that keep the layer's output on calibration inputs as close as possible, not weights that are individually closest.

For a layer with weights W and calibration inputs X, the target is the squared difference between X·W and X·Ŵ. Its second derivative with respect to a row of weights is H = 2XᵀX, the same for every row. H says which directions in weight space the inputs can see. An error in a direction where H is small barely changes the output.

## Starting point: Optimal Brain Quantization {#obq}
> OBQ quantizes one weight at a time and, using the inverse Hessian, adjusts all not-yet-quantized weights to compensate for that weight's error.

OBQ comes from the Optimal Brain Surgeon line of work. After fixing one weight to a grid value, it solves in closed form for the best update to the remaining weights, then removes that weight from the inverse Hessian. It picks the weight with the smallest error next. This works well but costs too much on big layers, because each row is handled separately with its own order.

## Three changes that make it scale {#insights}
> Use one fixed column order for every row, update in lazy blocks of columns, and compute the needed inverse-Hessian rows with a Cholesky decomposition.

First, the greedy order matters little on large layers, so GPTQ quantizes columns left to right for all rows at once, sharing one inverse Hessian. Second, it applies updates in blocks of 128 columns and only then touches the rest of the matrix, which makes the work friendly to GPU memory bandwidth. Third, repeatedly updating the inverse Hessian is numerically fragile, so GPTQ precomputes the rows it needs from a Cholesky factor of H⁻¹, with a small damping term (a percentage of the mean diagonal) added to H first. Your `gptq` in 3.4 is this loop without the blocking.

## Results {#results}
> OPT-175B and BLOOM-176B quantize to 3 or 4 bits in about four GPU hours with a small perplexity increase, where round-to-nearest falls apart.

At 4 bits the perplexity increase is small. At 3 bits round-to-nearest collapses while GPTQ stays usable. The 175B model then fits on a single 80 GB GPU. With custom kernels that dequantize on the fly, the paper also reports generation speedups over fp16, since decoding is limited by memory traffic. Grouping, a separate scale per block of weights, pushes even lower bit widths further.

## Calibration and practical details {#practice}
> A small set of generic text is enough: 128 random segments of 2048 tokens is the usual calibration set.

The calibration data only has to produce realistic inputs for every layer. Layers are processed in sequence, each receiving inputs that already passed through the quantized layers before it. Later implementations added two accuracy improvements, act-order (quantizing columns with larger Hessian diagonals first) and group sizes like 128. Both cost some speed, and together they are the common checkpoint format today.

## In the course {#course}
> 3.4 builds the damped Hessian and the column loop on a 4 × 8 layer, then runs GPTQ on Llama 3.1 8B.

On the lesson's layer GPTQ's weight error is higher than round-to-nearest's but its output error is about 7× lower. That's the objective at work: weights drift in directions the inputs don't see. The dead feature in the calibration batch shows why the damping term exists.
