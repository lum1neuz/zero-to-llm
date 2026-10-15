---
gist: The KV cache, not the weights, decides how many users fit on a GPU. PagedAttention stores it in small fixed blocks found through a table, the way an operating system pages memory, so almost none of it is wasted.
minutes: 17
---

## What this lesson is for {#why}
> In 5.1 you saw that serving well means keeping the batch big. What limits the batch is KV cache memory, and the usual way of reserving it wastes most of it.

Every user in the batch needs their own KV cache, and it grows by one token every step. A server can't know in advance how long each answer will get. The simple fix is to reserve room for the longest possible answer, but most of that room is never used, and memory that sits reserved can't hold another user.

**PagedAttention**, from the vLLM paper, fixes this with an idea borrowed from operating systems: cut the memory into small fixed-size **blocks**, hand them out only as a sequence grows, and keep a **block table** per request that says where its blocks are. In this lesson you:

1. Work out what one token's KV cache costs, and how much contiguous reservation wastes.
2. Map token positions to blocks through a block table, and read a request's keys back in order.
3. Run attention over the gathered blocks and check it equals ordinary attention exactly.
4. On the GPU, page your 8B model's cache and measure how many more requests fit.

## What you need from before {#before}
> The KV cache stores one key and one value vector per layer, per KV head and per token, so the next token can attend to all of them without recomputing.

From lesson 2.2: during decode, each new token computes a query, then attends to the keys and values of every earlier token. Recomputing those keys and values every step would be wasteful, so they're kept in the **KV cache**. Each step adds one token's worth, in every layer.

From 1.1: attention for one query `q` over cached keys `K` and values `V` is

```
weights = softmax(K @ q / sqrt(d))
output  = weights @ V
```

with `K` and `V` of shape `[tokens, d]`, `q` of shape `[d]`, and the output of shape `[d]`. In this lesson nothing about that math changes. Only *where the rows of `K` and `V` are stored* changes.

From 5.1: your model's fp16 weights take about 15 GiB of a 24 GB card.

## How big is the KV cache? {#kv-size}
> Bytes per token = 2 × layers × kv_heads × head_dim × bytes per value. For Llama 3.1 8B in fp16 that's 128 KiB per token.

Count what one token leaves behind:

- a **key** and a **value**: that's the 2,
- in every **layer**: 32,
- for every **KV head**: 8,
- each a vector of **head_dim** numbers: 128,
- each number taking **2 bytes** in fp16.

```
2 × 32 × 8 × 128 × 2
  = 131,072 bytes
  = 128 KiB per token
```

A 4,096-token conversation needs 4,096 × 128 KiB = 0.5 GiB. Your model runs on about 4 GiB of KV memory, which is 32,768 tokens in total, shared by everyone in the batch.

**kv_heads, not heads.** Llama 3.1 8B has 32 *query* heads but only 8 *KV* heads. This is **grouped-query attention**: every 4 query heads share one key/value head. Only keys and values are cached, so only the 8 count. Size the cache with 32 and you reserve 512 KiB per token, 4× too much, and 4× fewer users fit. It's the most common mistake in KV arithmetic, and the first challenge checks for it.

## Reserving the maximum wastes most of it {#contiguous}
> If every request reserves max_len slots when it arrives, the memory between its real length and max_len sits empty and can't be lent to anyone else.

The simple scheme keeps each request's cache in one **contiguous** chunk, like one array per request. Arrays can't grow in place, so the chunk has to be sized for the worst case up front. Draw 3 requests with `max_len = 8`, where `#` is a used slot and `.` is reserved but empty:

```
req A  # # # . . . . .   3 of 8
req B  # # # # # # . .   6 of 8
req C  # . . . . . . .   1 of 8
```

10 of 24 slots hold tokens, so 58% is waste. Real numbers are worse. In the lesson's first challenge, 8 requests holding 37 to 3,010 tokens each reserve 4,096 slots apiece. 79.3% of the reservation sits empty.

There's a second problem. When requests finish, they leave holes of different sizes, and a new request needs one hole big enough for its whole maximum. Free memory split into pieces too small to use is called **external fragmentation**. A web developer has seen the same thing in a cramped CSS grid: plenty of empty cells, none in a block big enough for the next item.

## Pages: borrowing an idea from the operating system {#pages}
> Cut KV memory into fixed blocks, give a request a new block only when its last one is full, and keep a block table that maps its logical blocks to physical ones.

Your operating system already solves this problem for RAM. A program sees one long, continuous address space, its **virtual memory**. Behind the scenes that space is cut into 4 KB **pages**, and a **page table** says which physical frame holds each page. The frames can be anywhere.

PagedAttention does the same for the KV cache:

| Operating system | PagedAttention |
|---|---|
| Page (4 KB) | Block (16 tokens in vLLM) |
| Physical frame | Physical block in the pool |
| Page table | Block table, one per request |
| Virtual address | Token position |

All requests share one **pool** of physical blocks. A request's **block table** is a short list: entry 0 is the physical block holding its tokens 0–15, entry 1 the block holding 16–31, and so on. Those blocks can sit anywhere in the pool, in any order.

When a request fills its last block, the engine takes any free block off a **free list** and appends it to the table. When a request finishes, its blocks go back on the free list. Every block is the same size, so any free block fits any request, and external fragmentation disappears.

## From a position to a slot {#slots}
> To find token position pos: divide by the block size to get the logical block, look that up in the table to get the physical block, and the remainder is the offset inside it.

Small blocks make this easy to follow. Say blocks hold 4 tokens, and a request's block table is `[3, 0, 5]`:

```
logical block   0     1     2
positions      0-3   4-7   8-11
table entry     3     0     5
```

Where does position 6 live?

```
6 // 4 = 1   logical block 1
table[1] = 0 physical block 0
6 %  4 = 2   offset 2 in it
```

So position 6 is slot 2 of physical block 0. Two mistakes the tests look for: forgetting the lookup (answering "block 1", the *logical* number), and using `pos` itself as the offset. A block only has slots 0–3, so an offset of 6 can't exist. Using `/` instead of `//` fails too: `6 / 4` is `1.5`, and a list can't be indexed with a float.

## How much do blocks waste? {#block-waste}
> Each request rounds its length up to whole blocks, so it wastes at most one partly filled block: about half a block on average.

With blocks of 16, a request holding 37 tokens needs 3 blocks, which is 48 slots. Its first two blocks are full, and the third holds 5 tokens and 11 empty slots. That's the only waste. For the same 8 requests that wasted 79.3% contiguously, paging wastes **1.0%**.

Getting the count right takes care:

| Rounding | 37 tokens | 16 tokens | Problem |
|---|---|---|---|
| Up, per request | 48 | 16 | none, correct |
| Down (`37 // 16`) | 32 | 16 | drops 5 tokens |
| Always +1 block | 48 | 32 | wastes a block when the last is full |
| Up, on the total | n/a | n/a | treats blocks as shared |

The last row is subtle. Rounding up the *sum* of all lengths pretends requests can share a block. They can't: a block belongs to one request's table.

Block size is a trade-off. Bigger blocks mean shorter tables and bigger, more efficient reads, but about half a block of waste per request. At 16 tokens that's about 8 slots. At 256 it's about 128. vLLM defaults to 16.

## Reading a request's keys back {#gather}
> Pick the request's blocks from the pool in table order, lay them end to end, and cut off the empty tail of the last block.

The pool has shape `[num_blocks, block_size, d]`. To get one request's keys as an ordinary `[length, d]` tensor, follow the shapes:

```
pool              [6, 4, d]
its blocks, in
  table order     [3, 4, d]
end to end        [12, d]
first length rows [10, d]
```

Each step has a classic mistake:

- **Reading blocks 0, 1, 2 in pool order** instead of the table's order. Other requests own those blocks.
- **Keeping the blocks separate** (`[3, 4, d]`). Attention needs one row per token.
- **Not cutting to the length.** The last block's spare slots aren't zeros. They hold stale keys from whichever request used the block before, so attention would mix in someone else's past.
- **Hard-coding d = 2** because the script's example uses it. Read d from the pool's shape.

## Same math, different storage {#same-math}
> Gathering only moves numbers, so attention over the gathered blocks equals dense attention exactly, and the lesson's check prints a difference of 0.

Once a request's keys and values are back in order, attention is the formula from 1.1, unchanged. Your browser challenge compares the paged result with PyTorch's own attention over the original keys and values. A difference of 0 is the healthy answer.

Real engines skip the copy. vLLM's kernel loops over a request's blocks and reads each one in place, using the table, while it computes. That saves memory traffic but sums things in a different order, so on the GPU you'll see tiny fp16 rounding differences instead of 0. The challenge's mistakes show up as large differences instead: forgetting the 1/√d scale, dividing by d instead of √d, taking values from the key pool, or skipping the softmax.

A bonus of block tables, used in 5.4: two requests that start with the same prompt can point their tables at the *same* physical blocks and share them.

## The GPU run {#gpu-run}
> You'll swap your 8B model's cache for a paged one, check the logits barely move, then replay 200 real requests to measure the waste.

The course repo's `paged_kv.py` runs two checks, in about 40 minutes:

1. **Does paging change the output?** It generates from 20 prompts with the model's own cache and with a paged cache of 16-token blocks, then compares. Healthy: per-layer differences around 1e-4 (fp16 rounding), a largest logit difference of a few thousandths, and identical greedy tokens. A difference near 1 means a slot is being read from the wrong place.
2. **How much memory does it save?** It replays 200 Mongolian chat requests and counts reserved-but-empty slots. Healthy: contiguous reservation at 4,096 tokens wastes about 80%, paging about 1%, and the number of requests that fit in 4 GiB jumps from 8 to about 46.

The lesson's replay of this run is **illustrative sample data written for the course, not a real recording**. You record the largest logit difference and the paged waste percentage from your own run.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **KV cache**: the stored key and value vectors of every earlier token, per layer, reused at every decode step.
- **kv_heads**: how many key/value heads each layer has (8 for Llama 3.1 8B). This is what the cache size depends on.
- **Query heads (heads)**: how many attention heads compute queries (32 for Llama 3.1 8B).
- **Grouped-query attention (GQA)**: several query heads sharing one K/V head, which shrinks the cache.
- **head_dim**: the length of one head's key or value vector (128).
- **KiB, GiB**: 1,024 bytes and 1,024³ bytes.
- **max_len**: the longest sequence a request may reach. Contiguous schemes reserve this much up front.
- **Contiguous**: stored in one unbroken chunk.
- **Reserved but empty (waste)**: KV slots set aside for a request that hold no token.
- **Fragmentation**: free memory split into pieces too small or scattered to use.
- **Block**: a fixed number of token slots (16 in vLLM) in the KV pool.
- **Pool**: all physical blocks, shared by every request.
- **Block table**: one request's list of physical blocks, in logical order.
- **Logical block**: a block's position within one request (0, 1, 2…).
- **Physical block**: where that block actually sits in the pool.
- **Slot, offset**: one token's place inside a block, numbered 0 to block_size − 1.
- **Gather**: read a request's scattered blocks back into one ordered tensor.
- **Free list**: the blocks not owned by any request, ready to hand out.
- **Dense attention**: ordinary attention over keys and values stored in one tensor.
- **Stale data**: leftovers in a reused slot from an earlier request.
- **Logit difference**: the largest gap between two runs' output scores. This lesson's quality metric.
- **Illustrative**: sample data written for the course, not a recording of a real run.
