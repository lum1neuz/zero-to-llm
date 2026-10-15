---
paper: Efficient Memory Management for LLM Serving with PagedAttention
gist: Storing each request's KV cache in small fixed-size blocks, found through a per-request block table the way an OS pages memory, removes most of the KV memory waste and lets a server batch several times more requests.
---

## The problem {#problem}
> In LLM serving, the KV cache, not the weights, limits how many requests fit on a GPU, and earlier systems wasted most of it.

Each request's KV cache grows by one token per decode step and can reach gigabytes for long contexts. The final length isn't known in advance. Earlier serving systems kept each request's cache in one contiguous chunk, so they had to reserve room for the maximum possible length up front. The paper's measurements show that only a minority of the KV memory in those systems held actual token states. The rest was reserved but empty, or lost to fragmentation. Unused KV memory means fewer requests in the batch, and a smaller batch means lower throughput.

## Three kinds of waste {#waste}
> Over-reservation, internal fragmentation and external fragmentation all come from insisting that a request's KV cache be contiguous.

Reservation waste: slots set aside for tokens that haven't been generated yet, and may never be. Internal fragmentation: a request that stops early never uses the end of its chunk. External fragmentation: the gaps between chunks of different sizes are too small to reuse. You measure the first two in 5.2's `kv_budget` challenge, where contiguous reservation wastes about 79% and paging about 1%.

## PagedAttention {#pagedattention ref="§4.1"}
> Split each request's KV cache into fixed-size blocks that can live anywhere in GPU memory, and make the attention kernel read them through a block table.

A block holds the keys and values of a fixed number of tokens (16 by default in vLLM). Logical block i of a request maps to some physical block through that request's block table, just as virtual pages map to physical frames. The attention kernel loops over the request's blocks, fetches each one from wherever it is, and computes the same result as attention over a contiguous cache. That's 5.2's `block_table` and `paged_attention` challenges. You gather the blocks first, while the real kernel reads them in place.

## Allocating on demand {#allocation ref="§4.2"}
> A request gets a new physical block only when its last one fills, so its waste is at most one partly filled block.

Because any free block can serve any request, there's no external fragmentation, and a central block manager just keeps a free list. Memory waste drops to the unfilled tail of each request's last block. That's why block size is a trade-off: bigger blocks waste more per request but mean fewer table entries and larger, more efficient reads.

## Sharing blocks {#sharing ref="§4.4"}
> Block tables make sharing cheap: several sequences can point at the same physical block, with reference counts and copy-on-write.

Parallel sampling and beam search produce several sequences that share a prompt. With paging, all of them point at the prompt's blocks instead of copying them. When one sequence needs to write into a shared block, that block is copied first (copy-on-write). The same mechanism underlies prefix caching across requests, which 5.4 builds with chained block hashes.

## Scheduling and preemption {#scheduling ref="§4.5"}
> When blocks run out, vLLM preempts whole requests and either swaps their blocks to CPU memory or drops them and recomputes later.

The scheduler serves requests first-come, first-served. If generation needs more blocks than are free, the most recently arrived requests are evicted as a unit, because a sequence needs all its blocks to continue. Their KV can be swapped out to CPU memory or simply recomputed from the tokens when they resume. Recomputation is often cheaper than it sounds, because prefill processes all the tokens in parallel.

## What it achieves {#results}
> vLLM, the system built on PagedAttention, served about 2–4× the throughput of earlier systems at similar latency.

The gains come mostly from fitting more requests into one batch, since decode is memory-bound and a bigger batch is nearly free per step (5.1's roofline). The gap grows with longer sequences, larger models and sampling methods that share prompts.

## How the course uses it {#course}
> 5.2 builds the block table and gather-based attention and measures the waste; 5.4 adds the scheduler and prefix sharing that paging makes possible.

The browser challenges reproduce the core: KV bytes per token (with kv_heads), waste with and without blocks, slot lookup through a table, and attention over gathered blocks that matches dense attention exactly. The GPU replay swaps your 8B model's cache for a paged one. Lesson 5.4's continuous batching and prefix caching are the scheduler and sharing ideas from this paper and from the systems around it.
