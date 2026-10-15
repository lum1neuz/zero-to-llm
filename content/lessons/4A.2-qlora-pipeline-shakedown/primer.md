---
gist: QLoRA stores the frozen 8B base model in 4 bits and trains LoRA adapters on top, which brings a fine-tune down to about 10 GB. This lesson adds up where every gigabyte goes, then runs a short job to prove the pipeline works.
minutes: 15
---

## What this lesson is for {#why}
> You'll work out whether fine-tuning Llama 3.1 8B fits in 12 GB, and why, and then run a short test job before the long real one.

4A.1 showed that LoRA trains a tiny fraction of the parameters. But the frozen base model still has to sit in GPU memory, and Llama 3.1 8B in 16-bit numbers is 16 GB on its own. Your GPU has 12.

**QLoRA** combines two things you already know: the 4-bit quantization from phase 3 and the LoRA adapters from 4A.1. The base is stored in 4 bits and never changes. The adapters train on top in higher precision.

The lesson has two halves. First you do the memory arithmetic in the browser, so you know what each piece costs. Then you run a **shakedown**: a 35-minute job whose only purpose is to find the problems (out of memory, a loss that doesn't fall, a file that won't save) before you start 4A.4's 7-hour run. Think of it as a smoke test before a deploy.

## What you need from before {#before}
> Bytes per number decide memory: fp32 is 4, bf16 is 2, int8 is 1, 4 bits is half a byte. Each block of quantized weights also stores a scale.

From phase 3:

| Format | Bytes per number | Where you met it |
|---|---|---|
| fp32 | 4 | training by default |
| bf16 / fp16 | 2 | model weights for inference |
| int8 | 1 | 3.1 |
| 4-bit (int4, NF4) | 0.5 | 3.2, 3.5 |

A quantized block of weights shares one **scale** (its absmax), and that scale costs memory too. 3.2 counted it: with groups of 128 and fp16 scales, int4 really costs 4 + 16 / 128 = 4.125 bits per weight. **NF4** (3.5) is a 4-bit format whose 16 levels sit where normally distributed weights are most common.

From 4A.1: a LoRA adapter on a Linear layer has r × (in + out) trainable numbers, and every trainable number needs memory for its gradient and Adam's two averages.

In this course, **GB** means 10⁹ bytes, the same as 3.1. `nvidia-smi` and many PyTorch tools report GiB (2³⁰ bytes, about 7% bigger), so small differences are normal.

## Where training memory goes {#where}
> Six things share the GPU during a fine-tune: the weights, the adapters' training state, the saved activations, the logits, some working space, and the CUDA runtime itself.

| Part | What it is | Grows with |
|---|---|---|
| Base weights | the frozen model | model size, bits per weight |
| Adapter state | LoRA weights, gradients, Adam's m and v | rank r |
| Activations | values kept from the forward pass for backward | sequence length, batch |
| Logits | one score per vocabulary entry per token | sequence length, batch, vocabulary |
| Working space | one layer recomputed, temporary buffers | sequence length |
| CUDA context | the driver's own memory | fixed, about 0.5 GB |

The next sections put a number on each row.

## The 4-bit weights {#nf4}
> Each NF4 weight costs 4 bits plus its share of the block's scale. Double quantization shrinks that share from 0.5 to about 0.127 bits.

bitsandbytes, the library that does QLoRA's quantization, cuts the weights into blocks of 64. Each block keeps one **absmax**.

**Without double quantization**, each absmax is an fp32 number: 32 bits shared by 64 weights, which is 0.5 bits per weight. Every weight costs 4.5 bits.

**With double quantization**, the absmaxes are themselves quantized to 8 bits. Every 256 of those 8-bit absmaxes share one fp32 constant that brings them back to scale. Per weight:

| Piece | Bits per weight |
|---|---|
| the weight | 4 |
| its 8-bit absmax, shared by 64 | 8 / 64 = 0.125 |
| the fp32 constant, shared by 64 × 256 | 32 / 16,384 ≈ 0.002 |
| **total** | **≈ 4.127** |

For one billion weights that's 4.5 / 8 = 0.5625 GB without and about 0.516 GB with. The tiny third row is easy to forget, and the tests check for it.

Only the **Linear layers** inside Llama's 32 blocks get quantized, about 6.98 billion weights. The token embedding and the output head (`lm_head`) stay in bf16. Each is 128,256 × 4096, together 1.05 billion weights, or 2.1 GB at 2 bytes each. That's a third of the total:

| Version | Weight memory |
|---|---|
| everything bf16 | 16.06 GB |
| NF4, no double quant | 6.03 GB |
| NF4 + double quant | 5.70 GB |

## The adapters {#adapters}
> 41.9 million LoRA numbers at 16 bytes each come to 0.67 GB. That's all the training state there is, because the base gets none.

The shakedown puts r = 16 adapters on all seven Linear layers in each of Llama's 32 blocks. Here are their shapes. The key and value projections are narrow (1024 wide) because Llama shares 8 key-value heads between its 32 query heads:

| Layer | in → out | r × (in + out) at r = 16 |
|---|---|---|
| q, o | 4096 → 4096 | 131,072 each |
| k, v | 4096 → 1024 | 81,920 each |
| gate, up | 4096 → 14336 | 294,912 each |
| down | 14336 → 4096 | 294,912 |

One block's adapters add up to 1,310,720. Times 32 blocks, that's 41,943,040, or 0.52% of the model.

Each trainable number keeps four fp32 values while it trains: the weight, its gradient, Adam's m and Adam's v. 4 × 4 = 16 bytes, so 41.9M × 16 ≈ 0.67 GB. A classic mistake is counting only the weights (2 or 4 bytes). The optimizer state is most of the cost.

## Activations and gradient checkpointing {#activations}
> Backward needs values from the forward pass. Gradient checkpointing keeps only one hidden state per layer and recomputes the rest when needed.

To compute gradients, backward needs the intermediate values that forward produced: attention scores, the inputs to every matmul, and so on. Keeping all of them for 32 layers at 2048 tokens overflows 12 GB on the very first step. The shakedown's recording opens with exactly that crash.

**Gradient checkpointing** trades compute for memory. It keeps only each layer's input, one hidden state per token per layer:

```
32 layers × 2048 tokens
  × 4096 numbers × 2 bytes
≈ 0.54 GB
```

During backward it re-runs one layer's forward at a time to get the values it needs. That costs about a third more compute, and it's what makes the run fit. The budget allows about 1.6 GB for that one recomputed layer plus temporary buffers. That number is a rough estimate, not exact arithmetic.

## The logits surprise {#logits}
> Llama 3.1 scores every one of its 128,256 vocabulary entries at every position, and the loss works on those scores in fp32. For one 2048-token sequence that's 1.05 GB.

The last layer turns each token's 4096 hidden numbers into one score per vocabulary entry. These scores are the **logits**. The loss converts them to fp32, 4 bytes each:

```
2048 tokens × 128,256 entries
  × 4 bytes ≈ 1.05 GB
```

That's more than all 41.9M adapter numbers with their optimizer state. It also grows with the batch: four sequences at once would need 4.2 GB just for logits. That's why QLoRA runs one sequence at a time and uses **gradient accumulation**: run 4 sequences one after another, add up their gradients, then take one optimizer step. It's the same update as a batch of 4, with a quarter of the memory.

The tests check the classic mistakes here: counting 2 bytes per logit (the loss upcasts to fp32), forgetting the batch, and using the hidden size instead of the vocabulary size.

## Paged optimizers {#paged}
> If a step suddenly needs more memory than is free, a paged optimizer moves some of its state to CPU RAM instead of crashing.

Memory use isn't perfectly flat. Fragmentation, an unusual batch or an eval step can push one moment over the edge. A **paged optimizer** keeps Adam's state in CUDA's *unified memory*, which the driver can move between GPU and CPU memory on its own, like an operating system swapping pages to disk. A spike makes that step slower instead of ending the run with an out-of-memory error.

In the shakedown's recording, one logging window jumps to 11.8 GB, and the log says 0.4 GB of optimizer state was paged out. The run carries on.

## The whole budget {#budget}
> Add it all up and the fine-tune needs about 10.1 GB, under 2 GB to spare on a 12 GB card.

| Part | GB |
|---|---|
| NF4 base weights | 5.70 |
| LoRA weights, grads, Adam | 0.67 |
| logits (fp32, one sequence) | 1.05 |
| checkpointed activations | 0.54 |
| one layer recomputed (rough) | 1.60 |
| CUDA context | 0.50 |
| **total** | **10.06** |

Try doubling the sequence length to 4096 in the challenge. The logits and the checkpoints both double, and the total reaches about 11.6 GB before the recomputed layer grows too. That's why the course trains at 2048.

## The shakedown run {#shakedown}
> 200 steps on short English instructions, packed into full-length sequences. Healthy means it fits, the loss falls, and the adapter saves and generates.

On your GPU you load Llama 3.1 8B in NF4, add r = 16 adapters (alpha = 32) and train for 200 steps. The data is 6,000 short Alpaca-cleaned examples, packed into 800 sequences of 2048 tokens (**packing** joins short examples end to end so no space is wasted on padding). It's one pass over the data, 1.64M tokens, about 35 minutes. The replay is an **illustrative** recording written for the course, not a real run. What healthy looks like:

- The first attempt, without gradient checkpointing, runs out of memory. That's expected.
- With checkpointing and a paged optimizer, the base loads at 5.70 GB and the log shows 41,943,040 trainable parameters (0.52%).
- Training loss falls from about 1.4 to 1.1, and the eval loss on 200 held-out examples ends near 1.12.
- Peak reserved memory is about 9.9 GB (nvidia-smi shows a bit more, about 10.4 GB). Throughput is about 800 tokens per second.
- The adapter saves as a file of about 168 MB, and a sample answer reads sensibly.

One setup mistake to watch: `prepare_model_for_kbit_training` freezes every parameter it finds, so call it *before* adding the adapters. Called after, it freezes them too, and the loss sits flat from step one.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **QLoRA**: LoRA adapters trained on top of a frozen base model stored in 4 bits.
- **NF4 (4-bit NormalFloat)**: a 4-bit format whose 16 levels suit normally distributed weights.
- **FP4**: a plain 4-bit float format, the alternative NF4 is compared against.
- **Absmax / block**: each block of 64 weights shares one scale, its largest absolute value.
- **Double quantization**: storing those absmaxes in 8 bits, with one fp32 constant per 256 of them.
- **bitsandbytes**: the library that quantizes the model and provides the paged optimizers.
- **`BitsAndBytesConfig`**: the settings object for loading a model in 4 bits: quant type, double quant, compute dtype.
- **Compute dtype**: the precision the matmuls run in (bf16). Weights are expanded from 4 bits just before each one.
- **Optimizer state**: Adam's m and v, kept for every trainable number.
- **Activations**: values from the forward pass that backward needs.
- **Gradient checkpointing**: keeping only each layer's input and recomputing the rest during backward.
- **Logits**: the model's raw score for every vocabulary entry at every position.
- **Gradient accumulation**: adding up gradients over several small batches before one optimizer step.
- **Paged optimizer**: an optimizer whose state can move to CPU RAM under memory pressure.
- **`prepare_model_for_kbit_training`**: PEFT's setup step for a quantized model. It freezes the base and turns on checkpointing.
- **Packing**: joining short examples into full-length sequences so no tokens are spent on padding.
- **Shakedown**: a short end-to-end run that checks the pipeline before a long one.
- **Peak VRAM / `max_memory_reserved`**: the most GPU memory PyTorch's allocator held at once during the run.
- **Eval loss**: the loss on held-out examples, the lesson's quality metric.
- **GB vs GiB**: 10⁹ bytes vs 2³⁰ bytes. This course uses GB.
