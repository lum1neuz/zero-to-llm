---
paper: FlashAttention
gist: Exact attention can run much faster and use memory that grows only linearly with sequence length, if it's computed in tiles that stay in the GPU's fast on-chip memory instead of writing the full T × T score matrix out to GPU memory.
---

## Attention is limited by memory traffic {#io}
> Standard attention writes the T × T scores and probabilities to GPU memory and reads them back, and that traffic, not the arithmetic, is what makes it slow.

A GPU has a large but relatively slow main memory (HBM) and a small, much faster on-chip memory (SRAM). The usual implementation computes `Q Kᵀ`, writes it out, reads it back for the softmax, writes the probabilities, then reads them again to multiply by V. For long sequences those T × T tensors are huge, so most of the time goes into moving them. The paper's word for this is IO-aware: count reads and writes to slow memory, not just FLOPs.

## Tiling {#tiling}
> Split Q, K and V into blocks small enough to fit in on-chip memory, and compute attention one block pair at a time.

The kernel loads a block of keys and values, then a block of queries, computes their scores on chip, and accumulates the contribution to the output. The full score matrix never exists in GPU memory. Everything for one layer happens in a single fused kernel, so intermediate results don't take a round trip through HBM.

## Softmax one block at a time {#online-softmax}
> Keep a running maximum and a running sum for each query row, and rescale the partial output whenever a new block raises the maximum.

Softmax normally needs the whole row before it can normalize. The trick is to process the row in blocks while tracking the largest score seen so far and the sum of exponentials. When a later block brings a bigger maximum, the earlier partial results are rescaled to match. At the end, the result is exactly the softmax-weighted sum. Nothing is approximated, which is why your `attention` and SDPA agree to about 1e-7 in 2.5.

## Recomputing in the backward pass {#backward}
> Instead of storing the attention matrix for backpropagation, store the small per-row statistics and recompute the scores block by block when gradients are needed.

Training normally saves the T × T probabilities for the backward pass. FlashAttention saves only the output and the softmax statistics, and recomputes scores from Q and K during backward. That's more FLOPs, but fewer bytes moved, and since attention was memory-bound, it's still faster. It also means memory grows linearly with sequence length instead of quadratically.

## What it achieves {#results}
> Faster training and inference for the same exact result, and room for much longer sequences.

The paper reports wall-clock speedups on attention itself and end to end when training BERT and GPT-2 style models, plus the ability to train with far longer contexts than standard attention fits in memory. It also describes a block-sparse variant that skips blocks entirely. Later versions improved the work partitioning further, and PyTorch's `F.scaled_dot_product_attention` can dispatch to a FlashAttention kernel on supported GPUs.

## How the course uses it {#course}
> 2.5 checks that SDPA computes the same attention you wrote in 1.1; on the GPU, the speedup shows up in prefill, where T is large.

During decode with a KV cache there's one query per sequence, so the score matrix is tiny and there's little traffic to save. That's why the 2.5 replay shows SDPA barely moving decode tokens/s but cutting prefill attention time. Long-context serving in phase 5 depends on kernels like this.
