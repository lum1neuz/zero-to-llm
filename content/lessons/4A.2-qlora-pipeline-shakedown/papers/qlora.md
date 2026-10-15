---
paper: QLoRA
gist: Store the frozen base model in a 4-bit NormalFloat format, train LoRA adapters through it, and use paged optimizers to survive memory spikes. That's enough to fine-tune a 65B model on a single 48 GB GPU with no loss against 16-bit fine-tuning.
---

## The problem {#problem}
> Even with LoRA, a 16-bit frozen base is too big for one GPU once models reach tens of billions of parameters.

LoRA removes the gradients and optimizer state of the base weights, but the weights themselves still sit in memory at 2 bytes each. A 65B model needs over 130 GB just for that. Earlier 4-bit methods worked for inference but hadn't been shown to support fine-tuning without losing quality.

## 4-bit NormalFloat {#nf4}
> NF4 places its 16 levels at quantiles of a normal distribution, so weights that are roughly normal use every level about equally often.

Pretrained weights are close to zero-centred and roughly normal. A uniform 4-bit grid wastes levels out in the tails, where few weights live. NF4 spaces its levels so each one covers an equal share of a standard normal, then scales each block of 64 weights by its absmax. The paper argues this makes NF4 close to optimal for normally distributed data, and it measured better results than plain 4-bit floats (FP4).

## Double quantization {#double-quant}
> The per-block scales are themselves quantized to 8 bits, cutting their overhead from about 0.5 to about 0.127 bits per weight.

With blocks of 64 and an fp32 absmax each, the scales cost 32 / 64 = 0.5 bits per weight. QLoRA quantizes those absmaxes to 8 bits in blocks of 256, each block with one fp32 constant. That saves roughly 0.37 bits per parameter, about 3 GB on a 65B model. The arithmetic is the first challenge in 4A.2.

## Paged optimizers {#paged}
> Optimizer state lives in unified memory, so when a long batch spikes memory it pages out to CPU RAM instead of crashing the run.

Gradient checkpointing keeps activation memory low on average, but a single long sequence can still spike it. NVIDIA's unified memory lets pages move automatically between GPU and CPU. QLoRA puts the optimizer state there, so a spike costs a slower step rather than an out-of-memory error.

## Adapters on every layer {#all-layers}
> To match 16-bit full fine-tuning, LoRA had to go on every Linear layer, not just the attention projections.

With adapters only on the query and value projections, QLoRA fell short of full fine-tuning. With adapters on all linear layers, it matched it across the model sizes they tested. The rank mattered much less than where the adapters were. That's why 4A.2 budgets r = 16 on all seven Linears.

## What it achieves {#results}
> 4-bit QLoRA matched 16-bit full fine-tuning on standard benchmarks, and their Guanaco chat models fine-tuned on one GPU came close to ChatGPT on the chatbot benchmarks they used.

The memory saving made fine-tuning 65B models practical on a single 48 GB GPU, and 7B to 13B models on consumer cards. The authors also found that a small, high-quality instruction set (OASST1, about 9,000 examples) beat a much larger one for chatbot quality. They warned that the GPT-4-judged chatbot benchmarks they relied on have real limits. 4A.5's harness asks the same question in a smaller way: what are you actually measuring?

## How the course uses it {#course}
> 4A.2 budgets QLoRA's memory on Llama 3.1 8B and shakes the pipeline down; 4A.4 and 4A.5 run real jobs on it.

The `BitsAndBytesConfig` in the fill-in card is the paper's recipe: NF4, double quantization, bf16 compute. Your budget adds the parts the paper doesn't dwell on: the bf16 embedding and output head, and the fp32 logits of a 128k vocabulary.
