---
paper: "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
gist: A kernel's attainable speed is capped by either the machine's peak compute or its memory bandwidth times the kernel's operational intensity (FLOPs per byte), and plotting that cap shows which one you're up against.
---

## Operational intensity {#intensity}
> Divide the floating-point operations a kernel does by the bytes it moves to and from main memory: that ratio decides what limits it.

Every kernel does some arithmetic and moves some data. Operational intensity is FLOPs per byte of memory traffic. A matrix multiply on large square matrices does many operations per byte and has high intensity. A decode step at batch 1 does about 2 FLOPs per weight while reading 2 bytes per weight in fp16, roughly 1 FLOP per byte, which is very low.

## The roof {#roof}
> Attainable FLOP/s = min(peak FLOP/s, memory bandwidth × operational intensity).

Plot attainable performance against operational intensity on log-log axes. At low intensity the limit is a slanted line: bandwidth times intensity, so performance grows with intensity. At high intensity it's a flat line, the processor's peak. Together they look like a roof. A kernel's position under the roof tells you whether to work on its memory traffic or its arithmetic.

## The ridge point {#ridge}
> Where the slanted and flat lines meet is the minimum intensity a kernel needs to be able to reach peak compute.

The ridge point is peak FLOP/s divided by bandwidth. On an RTX 3090 that's about 71 TFLOPS / 936 GB/s, roughly 76 FLOPs per byte. Kernels left of it are memory-bound, and kernels right of it are compute-bound. The 2.5 calc card finds this as a batch size: below about 76 sequences, Llama 3.1 8B decode is memory-bound, because batching raises the FLOPs per weight read.

## Ceilings {#ceilings}
> Below the roof sit lower ceilings for missing optimizations, and each one tells you which fix would lift performance.

The paper adds ceilings for things a kernel might not be exploiting: on the compute side, vector instructions or instruction-level parallelism, and on the memory side, things like prefetching or memory placement. If a kernel sits under a memory ceiling, a compute optimization won't help, and the other way round. The point of the picture is to tell you which kind of optimization is worth your time.

## Testing it on real machines {#evaluation}
> The authors apply the model to several multicore processors of the time and a set of scientific kernels, and it explains their measured performance.

They build rooflines for different multicore architectures and place kernels such as sparse matrix-vector multiply, stencils and FFTs on them. The machines have quite different balances of compute and bandwidth, so the same kernel can be memory-bound on one and compute-bound on another. The simple bound turned out to be a good guide to where each kernel landed and what would speed it up.

## How the course uses it {#course}
> 2.5's `decode_step_ms` is a two-number roofline: the step takes as long as the slower of compute and memory.

Your function computes time instead of FLOP/s, but it's the same model: compute time from peak FLOPs, memory time from bandwidth, take the larger. It explains why batching (2.4) is nearly free at small batch, why quantization (phase 3) speeds up decode by reading fewer bytes, and why your tiny 30M model is bound by neither. Its profile shows kernel launch overhead, which the roofline doesn't model at all.
