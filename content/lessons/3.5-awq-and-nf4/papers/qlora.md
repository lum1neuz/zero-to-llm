---
paper: QLoRA
gist: Freeze a base model in a new 4-bit format, NF4, train LoRA adapters on top in bf16, and you can fine-tune a 65B model on one 48 GB GPU without losing quality against 16-bit fine-tuning.
---

## The goal {#goal}
> QLoRA makes fine-tuning of very large models fit on a single GPU by storing the frozen base model in 4 bits.

Full fine-tuning of a 65B model needs hundreds of GB of GPU memory. LoRA already trains only small adapter matrices, but the frozen base weights still had to sit in 16 bits. QLoRA quantizes the base model to 4 bits, backpropagates through it into the adapters, and keeps the quality of 16-bit fine-tuning.

## NormalFloat 4 {#nf4}
> NF4 puts its 16 levels at quantiles of a normal distribution, because pretrained weights are roughly normal, so each level gets an equal share of weights.

Each block of weights is scaled by its absmax into [−1, 1]. The levels come from quantiles of a standard normal, rescaled to that range, built so that zero is represented exactly. The paper argues this is information-theoretically optimal for normally distributed data. In practice it beats evenly spaced 4-bit integers and 4-bit floats. This is what you implement in 3.5's `nf4_quantize`.

## Double quantization {#double-quant}
> The per-block constants are themselves quantized to 8 bits, cutting their cost from 0.5 to about 0.127 bits per weight.

With blocks of 64 weights and an fp32 absmax per block, the constants cost 32 / 64 = 0.5 bits per weight. QLoRA quantizes those constants to 8 bits in blocks of 256, with one fp32 constant per block of constants: 8 / 64 + 32 / (64 · 256) ≈ 0.127 bits. The saving of about 0.37 bits per parameter is around 3 GB on a 65B model.

## Paged optimizers {#paged}
> Optimizer states can be paged between GPU and CPU memory with NVIDIA unified memory, which absorbs the memory spikes of long sequences.

Gradient checkpointing produces short memory spikes on long batches that would otherwise run out of memory. Paged optimizers let those states spill to CPU memory and come back automatically. It's an engineering piece, but it's part of what makes single-GPU training reliable.

## How training works {#training}
> Weights are stored in NF4 and dequantized to bf16 on the fly; gradients flow through the frozen base into LoRA adapters on every linear layer.

Storage and compute types differ. NF4 is only the storage format, and each matmul runs in bf16 after dequantization. The paper finds that adapters on all linear layers, not just attention projections, are needed to match full 16-bit fine-tuning.

## Results {#results}
> 4-bit QLoRA matches 16-bit full and LoRA fine-tuning on the benchmarks tested, and the Guanaco models it produced were competitive chatbots.

Across model sizes, NF4 with double quantization recovers 16-bit fine-tuning performance. The paper's Guanaco family, trained on a small high-quality instruction set in about a day on one GPU, scored close to ChatGPT on the Vicuna benchmark under GPT-4 and human evaluation. The paper also argues that dataset quality matters more than size for instruction tuning.

## In the course {#course}
> 3.5 builds NF4 and does the double-quantization arithmetic; 4A.2 loads this format again to train adapters.

In 3.5 NF4 beats int4 absmax by about 28% on normally distributed weights at the same block size, and the calc cards work out 4.127 bits per weight and 5.7 GB for Llama 3.1 8B. The LoRA half of the paper comes back in 4A.1 and 4A.2, where you fine-tune on top of a model loaded in exactly this format.
