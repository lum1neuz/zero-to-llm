---
paper: AWQ
gist: A small share of weights matter far more than the rest, you find them by looking at activation sizes, and scaling their channels up before quantizing protects them while every weight stays low-bit.
---

## Not all weights are equal {#salient}
> Keeping about 1% of the weights in fp16 recovers most of the quality lost to int4, if those weights are chosen by activation magnitude.

The paper's starting experiment keeps a small fraction of weight channels in full precision and quantizes the rest. Chosen at random or by weight magnitude, this barely helps. Chosen by the size of the activations that feed them, it helps a lot. A channel with large inputs amplifies any error in its weights, so those are the weights to protect.

## Mixed precision is the wrong tool {#no-mixed}
> Keeping some weights in fp16 complicates storage and kernels, so AWQ looks for a way to protect salient channels with every weight still quantized.

A mixed-format weight matrix is awkward to store and slow to multiply. The goal becomes getting the same protection with a uniform low-bit format.

## Scaling protects salient channels {#scaling}
> Multiplying a salient weight channel by s > 1 before quantizing, and dividing its input by s, shrinks that channel's relative rounding error.

Rounding error per weight is roughly bounded by half a step, and the step is set by the group's largest weight. Scaling one channel up makes its weights larger relative to that step, as long as it doesn't become the group's maximum. Dividing the activation by the same s keeps the product unchanged. Push s too high and the scaled channel takes over the group's scale and hurts its neighbours, so there's an optimum.

## Searching the scale {#search}
> AWQ sets s = (mean |x| per channel)^α and grid-searches α per layer to minimise the layer's output error on calibration data.

A full optimisation over every channel's factor would be expensive and prone to overfitting. Tying all factors to activation magnitude through one exponent leaves a one-dimensional search, which is cheap. The paper also searches a clipping range for the weights. There's no backpropagation and no layer reconstruction like GPTQ's, which is why it's quick.

## Generalises well {#generalises}
> Because it fits so little to the calibration set, AWQ holds up when the calibration data differs from the deployment data, and works on instruction-tuned and multimodal models.

GPTQ's reconstruction can overfit to its calibration set. AWQ only measures average activation sizes, which are fairly stable across text domains. The paper shows it working on instruction-tuned and multimodal models as well as base LLMs.

## Fast inference {#inference}
> The authors pair AWQ with TinyChat, an inference system with fused 4-bit kernels that runs several times faster than fp16 Hugging Face code on desktop and mobile GPUs.

4-bit weights only pay off in speed if the kernels dequantize inside the matmul. The paper's system does that and targets edge devices as well as servers.

## In the course {#course}
> 3.5's `awq` does the scale-quantize-unscale step on an 8-channel layer and sweeps α by hand.

With two salient channels, α = 0.5 cuts output error more than 4× against plain rounding, and larger α overshoots. That's the trade-off the per-layer search handles. In the replay, AWQ on Llama 3.1 8B lands close to GPTQ.
