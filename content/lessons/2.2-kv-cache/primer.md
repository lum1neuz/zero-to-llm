---
gist: A token's key and value never change once computed, so a KV cache keeps them and each new token runs alone against the cache, which is fast but costs memory you can work out exactly with kv_heads.
minutes: 16
---

## What this lesson is for {#why}
> 2.1's loop redoes almost all of its work every step. A KV cache keeps the parts that never change, so each new token costs one position instead of the whole sequence.

In 2.1 your loop processed 2,010 positions to write 60 characters. Step 60 re-read all 63 tokens before it, even though 62 of them were exactly the same as in step 59. On a real model that waste is most of the generation time.

This lesson fixes it. You'll:

1. Write one **cached step**: run just the newest token through attention, using stored keys and values for everything before it.
2. Use it to **generate**, and check the text is identical to 2.1's.
3. Work out how much **memory** the cache costs for a real model, Llama 3.1 8B.

The cache is the foundation for the rest of the course's serving work. Batching (2.4), profiling (2.5) and the serving systems in phase 5 all revolve around it.

## What you need from before {#before}
> Attention compares each query with every key, turns the scores into weights with softmax, and uses them to mix the values. The causal mask stops a position from seeing later ones.

A refresher from 1.1 and 1.2. Each position's embedding `x` is multiplied by three weight matrices to give three vectors:

| Vector | Role |
|---|---|
| **q** (query) | "what am I looking for?" |
| **k** (key) | "what do I contain?" |
| **v** (value) | "what do I pass on if you pick me?" |

For one query, the steps are: dot it with every key, divide by √C (C is the width of the vectors), softmax over the keys, then take that weighted mix of the values. In the full forward this happens for every position at once, with a **causal mask** that sets scores for later positions to `-inf`, so they get weight 0.

A model with several **heads** does this several times in parallel, each head with its own narrower q, k and v.

## Where 2.1's time goes {#waste}
> When one token is added, only one new row of keys and values appears. Every old row is the same as last step.

Here's 2.1's loop from the attention layer's point of view:

```
step 59: compute k, v for
         positions 0 … 62
step 60: compute k, v for
         positions 0 … 63
```

Positions 0 to 62 get exactly the same keys and values in both steps. Only position 63 is new. And of all the outputs step 60 computes, only the last position's is used, because only the last row of logits predicts a new token.

So step 60 needs just two things: the new token's own q, k and v, and the keys and values of every earlier token, which already exist.

## Why old keys and values never change {#causal}
> Because of the causal mask, a token's key and value depend only on tokens at or before it. Adding a token later can't change them.

In the tiny model, a token's k and v come straight from its embedding plus its position embedding, so they obviously don't change. In a deep model, layer 2's input at position t is layer 1's output at position t, and the causal mask means that output only looked at positions 0…t. Follow that up through every layer and the conclusion holds: nothing at position t ever depends on a later token.

That's the guarantee that makes caching safe. It's like memoising a pure function in Java: the inputs for position t never change, so its result never needs recomputing.

## The cache, row by row {#cache}
> The cache is a table with one row per token: one table for keys, one for values. Each step adds one row to each.

Start with empty tables of shape `[0, C]`, zero rows of width C. Each step adds the new token's k as a new last row, and the same for v:

```
after token 0:  k cache [1, C]
after token 1:  k cache [2, C]
after token 2:  k cache [3, C]
…
after 7 tokens: k cache [7, C]
```

There's a shape detail here. A single key is a vector of shape `[C]`, but the cache is a table of shape `[t, C]`. `torch.cat` only joins tensors with the same number of dimensions, so the key has to become a one-row table, shape `[1, C]`, before it's joined underneath. Same idea as 2.1, where the new id needed a dimension before joining `idx`.

Two classic mistakes:

- **Replacing the cache instead of adding to it.** The cache stays at one row, so each new token only attends to itself and forgets the story.
- **Putting k into the value cache.** Copy-paste does this. The shapes still work, so nothing crashes, but the outputs come out wrong.

## One cached step {#step}
> The new token's query is scored against every cached key, softmaxed into weights, and used to mix the cached values: ordinary attention for a single query.

Here are the shapes for one step, with t earlier tokens already cached:

| Tensor | Shape |
|---|---|
| new token's q | `[C]` |
| key cache (with the new k) | `[t + 1, C]` |
| scores, one per cached key | `[t + 1]` |
| weights after softmax | `[t + 1]` |
| value cache | `[t + 1, C]` |
| mixed output | `[C]` |

A tiny worked example with C = 2 and three cached keys:

```
q    = [1, 0]
keys = [1, 0]
       [0, 1]
       [1, 1]
dots = [1, 0, 1]
÷ √2 = [0.71, 0, 0.71]
softmax ≈ [0.40, 0.20, 0.40]
```

The output is 0.40 × value 0 + 0.20 × value 1 + 0.40 × value 2.

**No mask needed.** The cache only holds the new token and the ones before it. There's no future to hide.

The tests compare your cached logits with the full forward at every position, and they recognise the usual slips by the size of the error: forgetting to divide by √C, dividing by C instead, or skipping softmax and mixing the values with raw scores.

## Prefill, then one token at a time {#prefill}
> First push the prompt through to fill the cache (the prefill). Then each new token is one step, at the position it actually sits in.

Generation with a cache has two phases:

1. **Prefill.** The prompt's tokens go through the cache, so their keys and values are stored. In the browser you do this one token at a time. On a GPU the whole prompt goes through in one parallel pass, which is much faster.
2. **Decode.** Pick the next token from the last logits, append it, and run *only that token* through one cached step.

Positions are where off-by-one bugs live. With a 4-token prompt, the tokens sit at positions 0, 1, 2 and 3. The first new token sits at position 4. Once you've appended it, `idx` is 5 long, so its position is the length minus one, not the length:

| After appending | len(idx) | New token's position |
|---|---|---|
| 1st new token | 5 | 4 |
| 2nd new token | 6 | 5 |

Get it wrong and every new token gets the next position's embedding, and the text drifts away from 2.1's.

With the cache, 60 new characters from a 4-character prompt cost 64 positions of work instead of 2,010. The tests check that number too: if something still re-runs the whole sequence, the count gives it away even when the text is right.

## Why Q isn't cached {#no-q}
> An old token's query was only ever used to compute that token's own output. Its key and value are read by every later token.

Each step needs the newest token's query and *every* earlier key and value. Old queries are never read again. Caching them would cost memory and buy nothing. That's why it's a **KV** cache.

## How big is the cache? {#size}
> Bytes = 2 × layers × kv_heads × head_dim × context × bytes per value. For Llama 3.1 8B at 8k context in fp16 that's exactly 1 GiB per sequence.

Count what's stored. For every **layer**, every **KV head**, and every **position**, there's one key vector and one value vector, each **head_dim** numbers long, each number taking some **bytes**.

Llama 3.1 8B's config says 32 layers, 32 attention heads, 8 key/value heads and a hidden size of 4096. The head width is the hidden size split across the query heads: 4096 ÷ 32 = 128.

| Factor | Value | Running total |
|---|---|---|
| K and V | 2 | 2 |
| layers | 32 | 64 |
| KV heads | 8 | 512 |
| head_dim | 128 | 65,536 |
| context | 8,192 | 536,870,912 |
| bytes (fp16) | 2 | 1,073,741,824 |

1,073,741,824 bytes is exactly 2³⁰, which is **1 GiB**. (A GiB is 2³⁰ bytes; a GB is 10⁹. The lesson uses GiB.)

A batch of 16 sequences at 8k needs 16 GiB, about as much as the fp16 weights. At Llama 3.1's full context of 131,072 tokens, one sequence needs 16 GiB on its own.

## kv_heads, not heads {#gqa}
> Llama 3.1 uses grouped-query attention: 4 query heads share each key/value head, so the cache holds 8 heads per layer, not 32. Counting query heads gives 4 GiB, which is wrong.

The most common mistake with this formula is using the number of attention heads. That counts a key and value for every query head:

```
32 heads → 4 GiB  (wrong)
 8 heads → 1 GiB  (right)
```

In older models every query head had its own key and value head, so the two counts were equal and the mistake was invisible. Newer models use **grouped-query attention (GQA)**: query heads are split into groups, and each group shares one key/value head. Queries are never cached, so only the key/value heads count. The config entry that counts them is called `num_key_value_heads`.

A related slip gives the same 4 GiB: computing the head width as hidden size ÷ KV heads (4096 ÷ 8 = 512). The width comes from the *query* heads: 4096 ÷ 32 = 128. The tests can't tell these two apart from the number alone, so the message mentions both.

For comparison, your 30M model has 6 layers and 8 heads of width 64, with no grouping. At 512 tokens in fp32 its cache is 2 × 6 × 8 × 64 × 512 × 4 bytes = 12 MiB.

## On your GPU {#gpu}
> You'll add a cache to your 30M model and compare it with 2.1's loop. In the recording, time per token stays flat as the story grows. It's illustrative sample data, not a real run.

The replay shows `generate.py --compare-cache` writing 512 tokens both ways. Healthy output looks like:

- With the cache, about 2 ms per token at every position. Without it, the time climbs from about 3 to 7 ms.
- Overall speed going from about 216 to about 480 tokens per second.
- The same 512 tokens both ways, with logits differing by around 2 × 10⁻⁶ in fp32.

The numbers are sample data, labelled as such in the app. In fp16, differences around 10⁻³ are normal; in fp32, anything above 10⁻⁴ means a bug. Compare layer by layer to find it.

On the GPU the cache has two more dimensions: `[B, kv_heads, T, head_dim]`, with new rows joining along the time axis. Pass each step the absolute position of its token (often called `start_pos`), or every new token gets position 0's embedding.

You record two metrics: the largest logit difference between cached and uncached runs, and the Llama 3.1 8B cache size at 8k from your function. It should be 1 GiB, not 4.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **KV cache**: the stored keys and values of every earlier token, one table per layer for each.
- **Query, key, value (q, k, v)**: the three vectors attention makes from each token's embedding.
- **Causal**: a position only depends on positions at or before it.
- **Cached step**: running one new token through attention against the cache.
- **Prefill**: running the prompt through the model to fill the cache.
- **Decode**: generating new tokens one at a time after the prefill.
- **Position**: where a token sits in the sequence, counting from 0. It picks the position embedding.
- **Head / head_dim**: one of the parallel attention computations, and the width of its vectors.
- **num_attention_heads**: the number of query heads.
- **num_key_value_heads (kv_heads)**: the number of key/value heads, the ones that are cached.
- **Grouped-query attention (GQA)**: several query heads share one key/value head.
- **Multi-query attention (MQA)**: every query head shares a single key/value head.
- **hidden_size**: the width of each token's embedding. head_dim = hidden_size ÷ num_attention_heads.
- **Bytes per value**: 4 for fp32, 2 for fp16.
- **GiB**: 2³⁰ bytes, 1,073,741,824.
- **Logit match (max abs diff)**: the largest difference between cached and uncached logits, this lesson's quality metric.
- **KV cache size per sequence**: this lesson's systems metric, in GiB.
