---
gist: A fast server decides who's in the batch every step, so finished requests leave and new ones start at once, and it computes a shared system prompt once and reuses it for every user.
minutes: 15
---

## What this lesson is for {#why}
> Two scheduler tricks turn paged KV memory into speed under load: continuous batching keeps every slot busy, and prefix caching skips work that many requests share.

In 5.1, vLLM pulled ahead of llama.cpp once 16 users arrived at once. In 5.2 you built the paged KV memory that lets a server hold many requests. This lesson adds the two scheduling ideas that put that memory to work:

1. **Continuous batching.** A simple loop runs a batch until its longest answer is done. A short answer that finished early still holds its place, and new arrivals wait. Continuous batching re-decides the batch **every step**.
2. **Prefix caching.** Most chat requests start with the same long system prompt. Without caching, the server computes its keys and values again for every user. With caching, it computes them once and every later request reuses the blocks.

In the browser you simulate a scheduler step by step and build the block hashing behind prefix caching. On the GPU you measure what caching does to time to first token on your vLLM server.

## What you need from before {#before}
> Batching runs several sequences in one forward pass; the KV cache lives in blocks found through a table; and prefill is the expensive part of time to first token.

- **Batching** (2.4): sequences stacked as rows of one tensor, padded to the same length, with an attention mask so padding is ignored. If you skipped 2.4, the course repo's padded-batching starter in 5.4's folder covers the mask code.
- **Blocks and block tables** (5.2): a request's KV cache is a list of 16-token blocks anywhere in a shared pool. A new request only needs free blocks, not one big contiguous chunk, which is what makes admitting it mid-run cheap.
- **Prefill and TTFT** (5.1): before the first token, the model runs over the whole prompt to fill the KV cache. For a long prompt that **prefill** dominates the time to first token.

## Static batching leaves slots idle {#static}
> In a static batch every request holds its slot until the longest one is done, and nobody new can join until the whole batch ends.

Picture a tour bus. It leaves when it's full, and it doesn't come back until the last passenger is done, even if everyone else finished hours ago. Draw 3 slots over 12 steps, where `#` means the slot made a token and `.` means it sat idle:

```
step    0 1 2 3 4 5 6 7 8 9 ...
a (12)  # # # # # # # # # # ...
b  (3)  # # # . . . . . . . ...
c  (2)  # # . . . . . . . . ...
```

Requests `b` and `c` finish by step 2, but their slots stay empty for 9 more steps, and the requests waiting behind them can't start. This is how a plain `generate` loop over a batch behaves.

## Continuous batching {#continuous}
> Every step, finished requests leave and waiting requests take their free slots. A slot is never idle while someone is waiting.

Now picture a restaurant instead of a bus. When a table frees up, the next party in line sits down. Nobody waits for the whole room to finish dinner.

This is also called **iteration-level scheduling**, from the Orca paper. vLLM, TGI and SGLang all work this way. It's cheap because of 5.2: a newcomer only needs a few free blocks, and a finished request's blocks go straight back on the free list.

On the challenge's 6 requests with 3 slots:

| Schedule | Last token at step | Mean latency |
|---|---|---|
| Static | 16 | 13.3 steps |
| Continuous | 11 | 4.8 steps |

Same requests, same GPU, same work per step. Only the scheduling changed.

## One step of the scheduler {#step}
> Each step does three things in order: admit waiting requests into free slots, make one token for everyone running, then retire whoever just finished.

A tiny trace with 2 slots. X wants 2 tokens and Y wants 1, both arriving at step 0. Z wants 2 and arrives at step 1:

| Step | Admit | Make a token | Retire |
|---|---|---|---|
| 0 | X, Y | X 1/2, Y 1/1 | Y done at 0 |
| 1 | Z (Y's slot) | X 2/2, Z 1/2 | X done at 1 |
| 2 | (none) | Z 2/2 | Z done at 2 |

With static batching, Z would wait for the [X, Y] batch to end at step 1, then run alone at steps 2 and 3.

Each part has a classic mistake the tests catch:

- **Admitting before arrival.** A request can only join once its arrival step has come.
- **Ignoring the batch limit.** Never more than `max_batch` requests running at once.
- **Admitting one per step.** If two slots are free, two requests can join in the same step.
- **Waiting for the batch to drain.** That's static batching again.
- **Retiring late.** Record the step in which the last token was made, and free the slot that same step.
- **Never freeing the slot.** Finished requests keep their seats, and nobody else ever gets in.

## Why a shared prompt is expensive {#prefill}
> Every request repeats the same system prompt, and prefill computes keys and values for all of it, every time.

A chat product usually puts a long **system prompt** in front of every conversation: instructions, a persona, a few worked examples. In the lesson's GPU run it's 1,214 tokens, followed by a user question of 40 to 180 tokens.

Without caching, each of 200 requests prefills all ~1,350 of its tokens: about 271,000 tokens of prefill in total, more than 88% of it identical work. That shows up directly in time to first token: about 470 ms at 16 users in the sample run.

The keys and values of those 1,214 tokens are the same for every request, because a token's K and V depend only on itself and the tokens before it. So compute them once and keep them.

## Prefix caching: reuse the blocks {#prefix}
> Keep finished requests' full KV blocks in a cache keyed by their content, and let a new request point its block table at matching blocks instead of recomputing them.

Think of a CDN or a browser cache. A response is stored under a key, the URL, and a later request with the same key gets the stored copy instead of hitting the origin server. Prefix caching does the same with KV blocks:

- **What's stored**: full 16-token blocks of keys and values.
- **The key**: a hash that identifies the block's tokens *and everything before them*.
- **A hit**: the new request's block table simply points at the cached physical block. Nothing is copied or recomputed.
- **Eviction**: when memory runs short, the least recently used cached blocks are freed.

Only **full** blocks are cached. A partly filled last block is still being written as the request grows, so it can't be shared yet.

## Chained hashes {#chain}
> A block's hash must cover its own tokens and its parent's hash, because the same tokens after a different past have different keys and values.

Why not just hash each block's 16 tokens? Because attention mixes in the past. Look at these two prompts with 4-token blocks:

```
P: [1 2 3 4] [5 6 7 8]
Q: [9 9 9 9] [5 6 7 8]
```

Block 1 holds `[5, 6, 7, 8]` in both, but in P those tokens attended to `1 2 3 4`, and in Q to `9 9 9 9`. Their keys and values differ. If block 1's hash only covered its own tokens, Q could reuse P's block and get the wrong past.

The fix is to **chain** the hashes. Each block's hash folds in the previous block's hash and its own tokens, starting from nothing:

```
h0 = hash(start,  block 0)
h1 = hash(h0,     block 1)
h2 = hash(h1,     block 2)
```

Now `h1` stands for everything from the start of the prompt to the end of block 1. Two prompts share `h1` exactly when their first two blocks match.

Two Python details the tests catch: a list can't be hashed (it could change), so turn each block into a tuple first. And hashing the *whole prompt* for every block is wrong the other way: then even block 0's hash changes whenever the user's question does, and nothing is ever reused.

## Looking up a new request {#lookup}
> Walk the new prompt's block hashes from the start, count block_size tokens for each hit, and stop at the first miss.

With the 1,214-token system prompt and 16-token blocks, the first 75 blocks hold 1,200 system tokens. Block 76 mixes the last 14 system tokens with the start of the user's question, so it differs per user and never matches. A new request reuses 1,200 tokens and prefills only the rest: its question plus those 14.

Stop at the first miss, even if a later block's hash is in the cache. That can happen after eviction: if block 1 was evicted but block 2 wasn't, block 2's keys and values were computed on top of a block 1 you no longer have, so you can't use it without recomputing block 1 first. The count is in **tokens**, not blocks: the caller needs to know how many prompt tokens to skip.

## The GPU run {#gpu-run}
> You'll benchmark your vLLM server at 16 users with prefix caching off, then on, and check that batching doesn't change greedy outputs.

The course repo's `bench_prefix.py` takes about 30 minutes. It sends 200 requests that share the 1,214-token system prompt, first with prefix caching off and then on, and prints TTFT, total prefill tokens and throughput. Then it compares 50 greedy outputs at 16 requests in flight against one at a time.

What healthy output looks like, from the lesson's replay. The replay is **illustrative sample data written for the course, not a real recording**:

- With caching on, about 88% of blocks hit, prefill drops from about 271,000 tokens to about 33,000, and TTFT p50 falls from about 470 ms to under 100 ms.
- Throughput rises a little, since less GPU time goes to prefill.
- About 48 of 50 greedy outputs match the batch-1 run. Batched kernels can round differently and flip a rare near-tie. Many mismatches would point to a bug, often in padding or attention masks.

The prefix-caching flag has changed between vLLM versions, and recent versions turn it on by default. Check `vllm serve --help`, and turn it off explicitly for the baseline run. You record the greedy match percentage and the TTFT with caching on.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Scheduler**: the part of a serving engine that decides which requests run in each step.
- **Static batching**: run a fixed batch until its longest request is done, then start the next.
- **Continuous batching (iteration-level scheduling)**: re-decide the batch every step, so requests join and leave freely.
- **Slot**: one place in the running batch, at most `max_batch` of them.
- **max_batch**: the most requests that may run at once.
- **Arrive**: the step at which a request reaches the server.
- **Admit**: move a waiting request into a free slot.
- **Retire**: remove a request that has made all its tokens and free its slot.
- **Latency (in steps)**: from arrival to last token, inclusive.
- **Drain**: wait until every request in the batch has finished.
- **System prompt**: the fixed text placed before every user's message.
- **Prefill**: the forward pass over the prompt that fills the KV cache.
- **TTFT (time to first token)**: from arrival to the first token. Mostly queueing plus prefill.
- **Prefix caching**: reusing the KV blocks of a prompt prefix seen before.
- **Full block**: a KV block whose every slot holds a token. Only these are cached.
- **Hash**: a number computed from data, used here as a cache key.
- **Chained hash**: a block hash that includes the previous block's hash, so it identifies the whole prefix.
- **Cache hit, miss**: the block is already cached, or it isn't.
- **Eviction**: freeing cached blocks when memory runs short, usually the least recently used first.
- **Chunked prefill**: splitting a long prompt's prefill across steps so others' decoding doesn't stall.
- **Illustrative**: sample data written for the course, not a recording of a real run.
