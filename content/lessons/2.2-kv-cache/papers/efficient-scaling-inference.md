---
paper: Efficiently Scaling Transformer Inference
gist: Serving a 540B-parameter model well means picking how to split it across chips for each phase and batch size, and shrinking the KV cache with multi-query attention so long contexts and big batches fit.
---

## Why inference is its own problem {#problem}
> Training runs huge batches that keep the chips busy, but generation produces one token per sequence per step, so it's limited by memory and communication rather than math.

The authors serve PaLM 540B on TPU v4 chips. They care about two different goals. One is latency, how fast a single user gets tokens back. The other is throughput, how many tokens the hardware produces per second, which sets the cost. These goals want different batch sizes and different ways of splitting the model, and the paper treats inference as an optimization problem of its own instead of reusing the training setup.

## Prefill and decode {#phases}
> Processing the prompt runs all its tokens in parallel; generating runs one new token at a time, reusing cached keys and values.

The prompt can go through the model in one big parallel pass, which keeps the chips busy. Each generated token then needs a full pass of its own. At small batch sizes that pass is dominated by reading the weights from memory, not by the math. That's the memory-bound decode step you measure in 2.5. The KV cache from this lesson is what makes each decode step cost one position instead of the whole sequence.

## The KV cache is a memory problem {#kv-memory}
> At long context and large batch, the cached keys and values can take more memory than the weights, and reading them every step costs time too.

Every layer keeps a key and a value for every past position of every sequence in the batch. The cache grows with batch size × context length, and each decode step reads all of it. That limits how big a batch fits in memory, and with it the throughput. Your `kv_cache_bytes` is the same accounting: 2 × layers × KV heads × head width × positions × bytes.

## Multi-query attention {#mqa}
> Sharing one key/value head across all query heads shrinks the cache by the number of heads, and the paper splits that cache across chips by batch.

With multi-query attention, every query head attends to the same single key and value head. The cache gets smaller by a factor of the head count. That frees memory for longer contexts and larger batches, and it cuts the bytes read per step. The paper also shows how to split that smaller cache across chips by batch instead of by head, so each chip doesn't hold a copy of the whole cache. Grouped-query attention, the next paper in this lesson, is the middle ground that Llama 3.1 uses.

## Partitioning the model {#partitioning}
> The best way to split weights and activations across chips depends on the batch size, so the paper derives cost models for several layouts and picks per phase.

For the feed-forward and attention weights, the paper compares layouts that keep weights in place and move activations with layouts that move the weights instead. Small batches favour keeping weights still, because activations are tiny. Large batches can afford to gather weights, because the activations are now the big thing to move. It's the same reasoning as the roofline arithmetic in 2.5: count what has to move and compare it with what has to be computed.

## Results {#results}
> With these choices and int8 weights, PaLM 540B generates with low per-token latency at small batch and keeps chip utilization high when processing prompts in large batches.

The paper reports low-latency generation at small batch sizes, helped by int8 weight quantization, which halves the bytes read per step. It also reports high hardware utilization when processing input tokens at large batch, with a 2048-token context. The broader lesson is that no single configuration wins: latency-oriented and throughput-oriented serving want different batch sizes and layouts.

## How the course uses it {#course}
> 2.2 builds the cache, 2.4 batches, 2.5 does the memory-vs-compute arithmetic, and phase 3 shrinks the bytes, the same levers this paper pulls at scale.

You won't partition a model across chips in this course, but every other idea here shows up. The KV cache and its size come from this lesson, batching from 2.4, the memory-bound decode argument from 2.5, and weight quantization from phase 3. Serving systems in phase 5 (PagedAttention, continuous batching) are about managing the same KV cache memory this paper worries about.
