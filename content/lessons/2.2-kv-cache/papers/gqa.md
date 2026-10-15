---
paper: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
gist: Grouped-query attention lets groups of query heads share one key/value head, getting most of multi-query attention's speed and memory savings with quality close to full multi-head attention, and an existing model can be converted with a little extra training.
---

## Two extremes {#extremes}
> Multi-head attention gives every query head its own keys and values; multi-query attention gives them all one shared set. One is accurate, the other is fast.

In standard multi-head attention, the KV cache holds one key and one value per head, per layer, per position. Multi-query attention (MQA) keeps all the query heads but uses a single key/value head, so the cache shrinks by the number of heads and each decode step reads far less memory. The cost is quality: with only one key/value head the model has less capacity, and training can be less stable.

## Grouped-query attention {#gqa}
> Split the query heads into G groups, and give each group one shared key/value head.

With G groups, G = 1 is multi-query attention and G = the number of heads is ordinary multi-head attention. Anything in between trades cache size for quality. Llama 3.1 8B uses 32 query heads and 8 key/value heads, so each group has 4 query heads. That's why your `kv_cache_bytes` multiplies by `num_key_value_heads` and gets 1 GiB at 8k context, not 4.

## Uptraining existing checkpoints {#uptraining}
> You don't have to train from scratch: merge each group's key and value heads by averaging them, then continue pretraining briefly.

The paper converts multi-head checkpoints into grouped (or multi-query) ones. For each group, the key projection matrices of its heads are mean-pooled into one, and the same for the values. The converted model is then trained further for a small fraction of its original pretraining compute so it can adapt to the shared heads. Mean-pooling worked better than alternatives such as keeping just one of the original heads.

## What it buys {#results}
> Uptrained GQA gets quality close to multi-head attention at speed close to multi-query attention.

On their encoder-decoder (T5) experiments, multi-query attention was clearly faster at inference but lost some quality. Grouped-query attention with a handful of groups recovered most of that quality while staying close to multi-query speed. The reason is the decode step from 2.5: generation is limited by bytes read per step, and the KV cache is a big part of those bytes once contexts get long.

## Why the group count matters for memory {#memory}
> The KV cache scales with the number of key/value heads, not query heads, so the group count directly sets how much cache each token costs.

Query heads are cheap at decode time: only the newest token's queries are computed, and they're never cached. Keys and values are stored for every past position. Cutting key/value heads from 32 to 8 cuts the cache 4×, which means 4× longer contexts or 4× bigger batches in the same memory. Counting query heads instead of key/value heads is the mistake the `kv_bytes` challenge catches.

```chart
kind: bar
title: Llama 3.1 8B's KV cache at 8k context (fp16)
x: { label: key/value heads per layer, values: [32 (one per query head), 8 (grouped)] }
y: { label: cache size (GiB), min: 0, max: 4 }
series:
  - { name: cache size (GiB), values: [4, 1] }
caption: Sharing each key/value head among 4 query heads cuts the cache from 4 GiB to 1 GiB per sequence.
```

## How the course uses it {#course}
> 2.2's cache-size formula uses kv_heads because of this paper; phase 5's serving systems are built around the same memory budget.

The tiny model in this lesson has a single head, so there's nothing to group. On your GPU, Llama-style models in phases 4A and 5 use grouped-query attention, and the cache you size here is what PagedAttention (5.2) and continuous batching (5.4) manage.
