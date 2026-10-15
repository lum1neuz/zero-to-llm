---
paper: "ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers"
gist: Post-training quantization of large transformers works much better with finer-grained scales (groups for weights, tokens for activations), a cheap layer-by-layer distillation step, and kernels that hide the quantization overhead.
---

## The problem {#problem}
> Quantizing a trained transformer without retraining tends to lose accuracy, and quantization-aware training is too expensive for large models.

Training a model with quantization in the loop recovers accuracy but needs the training data, lots of GPU time and a full training setup. Post-training quantization skips all that, but with coarse scales it damaged the BERT and GPT-style models the authors tried, especially at low bit widths. ZeroQuant aims to make post-training quantization accurate and cheap, and to actually run faster afterwards.

## Group-wise weight quantization {#group-wise}
> Split each weight matrix into groups and give every group its own scale, so one large weight can't coarsen the whole matrix.

This is the technique you build in this lesson. A single scale per matrix, or even per row, lets one large weight set the step size for many ordinary ones. Smaller groups keep the step size tied to the local range of the weights. The paper picks group layouts that match how GPU matmul kernels already tile the work, so the extra scales don't slow inference down.

## Token-wise activation quantization {#token-wise}
> Activations get one scale per token, computed on the fly, because their range varies a lot from token to token.

Weights are fixed, so their scales can be computed once. Activations change with every input, and a static scale chosen on calibration data either clips large tokens or wastes levels on small ones. Computing each token's range at run time fits every token, at the cost of a reduction per token, which the custom kernels absorb. The course quantizes weights only, so this part comes back when activations do.

## Layer-by-layer knowledge distillation {#lkd}
> Recover accuracy by training each quantized layer to match the original layer's output, one layer at a time.

Instead of fine-tuning the whole quantized model against a full-precision teacher, each layer is trained on its own to reproduce the original layer's output. Only one layer needs to be in training memory at a time, which keeps the cost low for large models. The paper reports that it still works without the original training data. This matters most at 4 bits and below, where plain rounding loses the most.

## The inference backend {#backend}
> Fusing quantization and dequantization into neighbouring operations is what turns smaller numbers into real speedups.

Quantized weights only save time if the extra steps are cheap. ZeroQuant fuses them into the surrounding kernels instead of running them as separate passes over memory. Without that work, quantization can save memory and still run slower than fp16.

## What it achieves {#results}
> int8 weights and activations with little or no accuracy loss on BERT and GPT-style models, and several-fold speedups over fp16 on their backend.

With the fine-grained scales and fused kernels, int8 models keep close to fp16 accuracy and run several times faster in the paper's benchmarks. Lower weight precisions, such as 4-bit weights with 8-bit activations, become usable once layer-wise distillation is added.

## How the course uses it {#course}
> 3.2 takes ZeroQuant's group-wise weight scales; activation quantization and distillation stay for later.

You quantize weights group-wise with absmax and dequantize before the matmul. There's no activation quantization, no distillation and no custom kernel. The group idea is the part that carries straight into every modern 4-bit format, from GPTQ checkpoints to llama.cpp's block quants in lesson 3.6.
