---
paper: LLM.int8()
gist: Large transformers can run matrix multiplications in int8 with no quality loss, if the few huge "outlier" feature dimensions stay in fp16.
---

## The problem {#problem}
> Halving weight memory with int8 is easy for small models, but naive int8 starts to hurt quality once models reach a few billion parameters.

fp16 needs two bytes per weight, so a 175B-parameter model needs about 350 GB just for weights. Storing weights as one-byte integers halves that. Earlier 8-bit methods worked on models of a few hundred million parameters. They lost accuracy as models grew, and the authors set out to find out why.

## Absmax and zero-point quantization {#absmax}
> Absmax maps the largest magnitude to ±127 with one scale; zero-point also shifts the range so asymmetric data uses all 256 levels.

Absmax is exactly what you build in 3.1: `scale = max|x| / 127`, then round `x / scale`. It's symmetric, so a row that is mostly positive wastes the negative half of the grid. Zero-point quantization adds an offset so the observed minimum and maximum map to the ends of the int8 range. It uses the grid better but makes the matmul arithmetic more complicated. Both share the same weakness: one extreme value sets the step size for everything it shares a scale with.

## Vector-wise quantization {#vector-wise}
> Give every row of the activations and every column of the weights its own scale, then undo both scales on the int32 result.

A matrix multiply is many inner products, one per (row of X, column of W) pair. If each row of X has its own scale and each column of W has its own scale, the int8 product of a pair can be rescaled with exactly two numbers. Rescaling the int32 output by the outer product of the row and column scales recovers the fp16 result. Finer scales mean a large value only hurts its own row or column, not the whole matrix.

## Emergent outlier features {#outliers}
> Around 6.7B parameters a phase shift happens: a handful of hidden dimensions carry very large values in almost every layer, and they wreck shared scales.

Outliers here are features with magnitude of 6 or more. In small models they come and go. Past roughly 6.7B parameters they become systematic: the same few hidden dimensions are large across most layers and most tokens. They are rare, but they matter. Removing them noticeably damages the model. With vector-wise scales, one of these dimensions inflates the scale for every row it appears in, so all the ordinary values in those rows collapse onto a few levels. You can see the same thing in 3.1's quiz, where a 40.0 weight in a row of small weights makes the small weights round to zero.

## Mixed-precision decomposition {#decomposition}
> Split the matmul in two: the few outlier columns run in fp16, everything else runs in int8, and the results are added.

Find the hidden dimensions where any activation exceeds the threshold. Multiply those columns of X with the matching rows of W in fp16, about 0.1% of the values. Quantize the rest vector-wise and multiply them in int8. Add the two partial results. The outliers keep full precision, and the int8 part no longer has them inflating its scales.

## What it achieves {#results}
> Models up to 175B parameters run with int8 weights and keep their fp16 perplexity and zero-shot accuracy, at about half the weight memory.

The paper shows no measurable degradation on large OPT and BLOOM models, which made it possible to run them on far less GPU memory. Speed is a separate question. The decomposition adds overhead, so small models can run slower than in fp16. The win is memory, not speed.

## How the course uses it {#course}
> 3.1 builds the absmax core on weights only; 3.3 comes back to the outlier problem and the decomposition idea.

In 3.1 you quantize weights per row with absmax and dequantize before the matmul. There's no activation quantization and no decomposition yet. The red bar in 3.1's replay is the same phenomenon on a small scale: one row whose maximum is far above the rest. Lesson 3.3 measures outliers properly and compares fixes, including this paper's mixed-precision split.
