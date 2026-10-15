---
paper: SmoothQuant
gist: Divide each activation channel by a factor and multiply the matching weights by the same factor, so outliers shrink and both sides fit int8 with plain per-tensor or per-token kernels.
---

## Goal: W8A8 without special kernels {#goal}
> SmoothQuant wants both weights and activations in int8 so the matmul runs on fast integer hardware, with no mixed-precision path.

Weight-only int8 saves memory but still multiplies in fp16. Quantizing activations too (W8A8) lets the GEMM itself run in int8, which saves both memory and time. LLM.int8() gets the quality but needs a slower fp16 side path. SmoothQuant aims for the same quality inside ordinary int8 GEMMs.

## Activations are the hard part {#hard-part}
> Weights are flat and easy to quantize; activations have a few channels with huge values, and those channels are consistent across tokens.

The paper's key observation is about where the outliers sit. They're concentrated in particular input channels, and a channel that is large for one token tends to be large for all of them. Within a channel the values are similar. So a per-channel scale for activations would fix it, but GEMM kernels can only apply scales along the outer dimensions, per token for X and per output channel for W, not along the inner dimension being summed.

## Migrating the difficulty {#migration}
> Since X @ W = (X / s) @ (s · W), dividing each input channel of X by s_j and multiplying the matching weights by s_j leaves the output unchanged.

The scale per input channel that kernels can't apply to X can be pushed into W instead, where it's applied once, offline. The activations get smoother, and the weights pick up a moderate per-channel factor. Weights have room to absorb it because they had no outliers to start with. This is the `smooth` function in 3.3.

## Choosing s with alpha {#alpha}
> s_j = max|X_j|^α / max|W_j|^(1−α), and α decides how much of the range moves from activations to weights.

With α = 0.5 the two sides end up with the same maximum for that channel, splitting the difficulty evenly. A larger α moves more into the weights, which helps models with especially strong activation outliers. The paper reports that around 0.5 works for most models, with somewhat larger values for the hardest ones. α = 0 changes nothing, and α = 1 moves all of it into the weights, which then become hard to quantize.

## Free at inference {#free}
> The factors come from a small calibration set and are folded into the preceding LayerNorm or linear layer, so inference pays nothing extra.

The maximum of each activation channel is measured once on calibration data. Dividing X by s can be merged into the parameters of whatever produced X, typically a LayerNorm's scale, and multiplying W by s is done once to the stored weights. After that the model is an ordinary W8A8 model.

## Results {#results}
> W8A8 with smoothing keeps accuracy close to fp16 on very large models, including OPT-175B and BLOOM-176B, and is faster and smaller than fp16.

Without smoothing, W8A8 on these models loses a lot of accuracy. With it, the gap mostly disappears. Because everything stays in int8 GEMMs, the paper reports real speedups and roughly halved memory compared with fp16 in its serving framework, not just memory savings.

## In the course {#course}
> 3.3 builds the same migration on a 4 × 6 example and compares it with LLM.int8()'s split.

On the lesson's tensors smoothing cuts int8 error about 9× without any fp16 path. The outlier feature still sets each token's scale, but it no longer dwarfs the other features. The replay compares naive W8A8, LLM.int8() and SmoothQuant on Llama 3.2 1B, and the metric you record is the SmoothQuant W8A8 perplexity change.
