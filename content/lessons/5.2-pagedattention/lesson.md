---
id: "5.2"
title: PagedAttention
summary: Size the KV cache properly, give every request a block table instead of one big reservation, and show attention over scattered blocks matches dense attention exactly.
phase: "5"
hours: 4
type: build
status: ready
protected: false
cut: null
prerequisites:
  - 4B.4
  - "2.2"
papers:
  - title: Efficient Memory Management for LLM Serving with PagedAttention
    authors: Kwon et al.
    year: 2023
    url: https://arxiv.org/abs/2309.06180
quality_metric:
  id: logits_match
  label: Paged attention matches dense attention
  unit: max abs diff
  note: Largest logit difference between your paged cache and the model's own cache, fp16, 20 prompts × 128 tokens. fp16 rounding gives small nonzero values. Anything near 1 means a slot is read from the wrong place.
  inputs:
    - { id: max_diff, label: Max abs logit diff, hint: From paged_kv.py check 1 }
  compute: value
  reference: 0.0039
  pass: { max: 0.02 }
systems_metric:
  id: kv_waste_pct
  label: KV memory wasted vs contiguous
  unit: "%"
  note: Reserved but empty KV slots with 16-token blocks, over 200 replayed requests. Reserving max_len = 4,096 per request wasted 82.9% of the same memory.
  inputs:
    - { id: waste_pct, label: Paged KV reserved but empty (%), hint: From paged_kv.py check 2 }
  compute: value
  reference: 1.1
  pass: { max: 5 }
quiz:
  - id: contiguous_waste
    q: A server reserves 4,096 KV slots for every request when it arrives. Requests end up holding 600 tokens on average. About how much of the KV memory sits empty?
    answer: 1
    options:
      - text: About 15%
        why: That's the share in use, 600 / 4,096. The rest is reserved for tokens that never come.
      - text: About 85%
        why: 1 − 600 / 4,096 ≈ 85%. The server had to reserve for the longest possible answer because it can't know how long each one will be.
      - text: None, because the server frees whatever a request doesn't use
        why: The reservation is one contiguous chunk per request. Freeing its tail would leave holes too small and scattered for other requests to use, which is the fragmentation paging removes.
  - id: block_size
    q: You raise the block size from 16 tokens to 256. What happens?
    answer: 2
    options:
      - text: Waste drops, because there are fewer blocks to manage
        why: Fewer blocks means bigger ones, and each request's last block is on average half empty. That half-block grows from 8 slots to 128.
      - text: Nothing changes except speed
        why: Speed changes a little (shorter tables, bigger reads), but memory changes more. Half a 256-token block per request adds up across 50 requests.
      - text: Each request wastes about 128 slots instead of 8, but block tables get 16× shorter
        why: Waste comes only from the partly filled last block, about half a block per request. Bigger blocks waste more and need fewer table entries. vLLM's default of 16 sits near the sweet spot.
  - id: kv_heads_quiz
    q: You size Llama 3.1 8B's KV cache with heads = 32 instead of kv_heads = 8. What goes wrong?
    answer: 0
    options:
      - text: Each token reserves 4× the memory it needs, so 4× fewer requests fit
        why: Only the 8 K/V heads are cached; 4 query heads share each one (grouped-query attention). Sizing by 32 reserves 512 KiB per token instead of 128.
      - text: Attention reads the wrong heads and outputs garbage
        why: The size of the reservation doesn't change which numbers attention reads. The model still works. It just runs out of KV memory 4× sooner.
      - text: Nothing, because the extra memory is never touched
        why: Untouched isn't free. Reserved KV memory can't go to other requests, so the server admits fewer of them.
---

## Goal

Build the core of **PagedAttention**. Work out what the KV cache really costs per token, then measure how much of it a contiguous reservation wastes. Split the cache into fixed blocks, map every token position to a slot through a **block table**, and run attention over the gathered blocks until it matches dense attention exactly. Then page your 8B model's KV cache on the GPU and measure the memory it frees.

## Learn

```card
type: concept
id: fragmentation
eyebrow: Why pages
title: Reserving for the longest answer wastes most of the memory.
visual:
  kind: bars
  max: 100
  bars:
    - { label: contiguous, value: 79.3, text: "79.3%" }
    - { label: paged, value: 1.0, text: "1.0%" }
body: |
  A request's KV cache grows by one token per step, and the server can't know where it will stop. Reserve `max_len` slots per request and most of them stay empty. **PagedAttention** borrows the idea of virtual memory from operating systems: KV memory is cut into blocks of 16 tokens, each request keeps a **block table** of the blocks it owns, and it gets a new block only when the last one fills. The bars show reserved-but-empty KV slots for 8 requests (612 to 3,010 tokens each, `max_len` 4,096).
```

```card
type: calc
id: kv_per_token
eyebrow: Do the math
title: How much KV cache does one token of Llama 3.1 8B take in fp16?
given:
  - { label: Layers, value: "32" }
  - { label: Query heads, value: "32" }
  - { label: KV heads, value: "8" }
  - { label: Head dim, value: "128" }
  - { label: Bytes per value, value: 2 (fp16) }
unit: KiB
answer: { min: 127.9, max: 128.1 }
placeholder: "0"
why_ok: "2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KiB per token. A 4,096-token request needs 0.5 GiB, which is why KV memory, not weights, limits how many users fit."
why_wrong: "Count a key and a value for each *KV* head in each layer: 2 × layers × kv_heads × head_dim × bytes. The 32 query heads share 8 K/V heads, so the cache only stores 8."
hint: Only K and V are cached, and there are only 8 of each per layer. 1 KiB is 1,024 bytes.
```

```card
type: predict
id: slot_lookup
eyebrow: Predict the output
title: Where does position 9 live?
code: |
  BLOCK = 4
  block_table = [7, 2, 9]
  pos = 9
  physical = block_table[pos // BLOCK]
  offset = pos % BLOCK
  print(physical * BLOCK + offset)
options:
  - text: "37"
    correct: true
    why: "9 // 4 = 2, so it's in logical block 2, which the table puts in physical block 9. 9 % 4 = 1, so it's slot 1 of that block: 9 × 4 + 1 = 37 in the flat pool."
  - text: "9"
    why: That's the logical position, where it would sit in a contiguous cache. The table moved block 2 to physical block 9.
  - text: "10"
    why: That adds the offset to the block *number*. Each block holds 4 slots, so block 9 starts at slot 36.
  - text: "36"
    why: That's the start of physical block 9. Position 9 is the second token in its block, so it sits one slot further.
hint: Find the logical block with //, look it up in the table, then find the slot inside that block with %.
```

```card
type: fill_blank
id: gather_blanks
eyebrow: Fill the blanks
title: Read one request's keys back out of the shared pool.
code: |
  def gather_kv(pool, block_table, length):
      blocks = pool[[[0]]]
      flat = blocks.reshape(-1, pool.shape[-1])
      return flat[:[[1]]]
tokens: [block_table, "range(len(block_table))", length, "len(block_table) * BLOCK"]
answer: [block_table, length]
tips:
  "range(len(block_table))": That reads physical blocks 0, 1, 2… in order, whoever owns them. The table says which blocks belong to this request.
  "len(block_table) * BLOCK": That keeps the empty slots at the end of the last block. They hold stale keys from an earlier request, and attention would attend to them.
why_ok: "Indexing with the table picks the request's blocks in logical order, the reshape lays them end to end, and the slice drops the unfilled tail of the last block."
why_wrong: "The pool is shared by every request. Which list says which blocks are this one's, and how many tokens does it actually hold?"
hint: The block table lists this request's physical blocks in order. The last block is usually only partly full.
```

```card
type: quiz
ref: contiguous_waste
```

```card
type: quiz
ref: block_size
```

```card
type: quiz
ref: kv_heads_quiz
```

## Build steps

### Size the KV cache

```challenge
id: kv_budget
file: kv_budget.py
minutes: 25
start: { code: "" }
setup:
  - say: "Here's Llama 3.1 8B's attention shape. Notice it has 32 query heads but only 8 KV heads: grouped-query attention."
    type: "# Llama 3.1 8B: 32 query heads share 8 KV heads (grouped-query attention)\ncfg = {\"layers\": 32, \"heads\": 32, \"kv_heads\": 8, \"head_dim\": 128}\n\n"
    pause: 2.6
  - say: "Two questions. How many bytes does each cached token cost? And how much reserved memory sits empty, if each request reserves max_len slots up front, or only the whole blocks it needs?"
    pause: 4.0
  - say: "Here's the stub, with 8 requests of very different lengths at the bottom. Fill in the three blanks."
    type_file: kv_budget/stub.py
    pause: 1.2
tests: kv_budget/tests.py
solution: kv_budget/solution.py
alternatives:
  - kv_budget/alternatives/math_ceil.py
  - kv_budget/alternatives/loop.py
  - kv_budget/alternatives/k_and_v.py
hints:
  nudge: For each token, every layer stores one key and one value per KV head. For the waste, ask how many slots each scheme sets aside before the tokens arrive.
  concept: "Bytes per token are 2 (K and V) × layers × kv_heads × head_dim × bytes per value. Use `kv_heads`: with grouped-query attention several query heads share one K/V head, and only K and V are cached. Contiguous: each request reserves `max_len`. Paged: each request reserves its length rounded *up* to a multiple of `block_size`, and `(n + block_size − 1) // block_size` counts its blocks."
  skeleton: |
    return _ * cfg["______"] * cfg["________"] * cfg["head_dim"] * _______________
    contiguous_reserved = ___(lengths) * _______
    paged_reserved = sum((n + block_size - _) // __________ * block_size for n in lengths)
  solution: |
    return 2 * cfg["layers"] * cfg["kv_heads"] * cfg["head_dim"] * bytes_per_value
    contiguous_reserved = len(lengths) * max_len
    paged_reserved = sum((n + block_size - 1) // block_size * block_size for n in lengths)
on_solve: "128 KiB per token, so a 4,096-token request reserves half a GiB. For these 8 requests, contiguous reservation leaves 79.3% of it empty and paging wastes 1.0%: at most one partly filled block per request. With 4 GiB of KV memory that's the difference between 8 users and nearly 40."
mistakes:
  - { file: kv_budget/mistakes/placeholders.py, fails: kv_per_token, says: "still returns `...`" }
  - { file: kv_budget/mistakes/heads.py, fails: kv_per_token, says: "Count `kv_heads`" }
  - { file: kv_budget/mistakes/no_two.py, fails: kv_per_token, says: "a key *and* a value" }
  - { file: kv_budget/mistakes/no_bytes.py, fails: kv_per_token, says: "not bytes" }
  - { file: kv_budget/mistakes/no_layers.py, fails: kv_per_token, says: "one layer's worth" }
  - { file: kv_budget/mistakes/hardcoded.py, fails: kv_per_token, says: "Read every number from `cfg`" }
  - { file: kv_budget/mistakes/contiguous_placeholder.py, fails: contiguous_waste, says: "`contiguous_reserved` is still `...`" }
  - { file: kv_budget/mistakes/contiguous_one.py, fails: contiguous_waste, says: "Each request reserves its own" }
  - { file: kv_budget/mistakes/contiguous_used.py, fails: contiguous_waste, says: "can't know how long" }
  - { file: kv_budget/mistakes/paged_floor.py, fails: paged_waste, says: "How do you round up?" }
  - { file: kv_budget/mistakes/paged_plus_one.py, fails: paged_waste, says: "only when there's a remainder" }
  - { file: kv_budget/mistakes/paged_pooled.py, fails: paged_waste, says: "as if requests shared blocks" }
  - { file: kv_budget/mistakes/paged_blocks.py, fails: paged_waste, says: "not slots" }
  - { file: kv_budget/mistakes/paged_used.py, fails: paged_waste, says: "filled its last block exactly" }
  - { file: kv_budget/mistakes/no_main.py, fails: script_runs, says: "are missing" }
```

### Map positions to blocks

```challenge
id: block_table
file: paged.py
minutes: 30
start: { code: "import torch\n\n" }
setup:
  - say: "Blocks of 4 tokens, small enough to see. This request's block table says its logical block 0 lives in physical block 3, block 1 in block 0, and block 2 in block 5."
    type: "BLOCK = 4   # tokens per block\n\n# This request's logical block 0 lives in physical block 3, block 1 in block 0, block 2 in block 5\nblock_table = [3, 0, 5]\n\n"
    pause: 3.0
  - say: "The pool is six physical blocks shared by every request. Here are this request's 10 keys, one per token, with d = 2."
    type: "# Six physical blocks, shared by every request, and this request's 10 keys (d = 2)\nk_pool = torch.zeros(6, BLOCK, 2)\nkeys = torch.randn(10, 2, generator=torch.Generator().manual_seed(0))\n\n"
    pause: 2.4
  - say: "Your turn. slot_of says where a position lives, and gather_kv reads a request's keys back in order. The loop writes the keys through slot_of and checks gather_kv returns them unchanged."
    type_file: block_table/stub.py
    pause: 1.2
tests: block_table/tests.py
solution: block_table/solution.py
alternatives:
  - block_table/alternatives/cat.py
  - block_table/alternatives/index_select.py
  - block_table/alternatives/divmod.py
hints:
  nudge: Each position raises two questions. Which of the request's blocks is it in, and where inside that block? The block table turns the first answer into a physical block.
  concept: "With `BLOCK` tokens per block, position `pos` is in logical block `pos // BLOCK`, at offset `pos % BLOCK`, and `block_table[logical]` is the physical block. To gather, index the pool with the whole table: `pool[block_table]` gives `[blocks, BLOCK, d]` in the request's order. `reshape(-1, d)` lays the blocks end to end, and `[:length]` drops the empty tail of the last block."
  skeleton: |
    physical = block_table[pos __ BLOCK]
    offset = pos _ BLOCK
    return pool[___________].reshape(__, pool.shape[-1])[:______]
  solution: |
    physical = block_table[pos // BLOCK]
    offset = pos % BLOCK
    return pool[block_table].reshape(-1, pool.shape[-1])[:length]
on_solve: "The keys went into blocks 3, 0 and 5 and came back as one [10, 2] tensor, identical to the original. That's the trick: the request sees one long sequence, while its memory sits wherever the allocator found free blocks. Its last block holds 2 tokens, and those 2 empty slots are all the memory this request wastes."
mistakes:
  - { file: block_table/mistakes/placeholders.py, fails: slot_of, says: "`physical` and `offset` are still `...`" }
  - { file: block_table/mistakes/float_div.py, fails: slot_of, says: "Which operator divides and drops the remainder?" }
  - { file: block_table/mistakes/per_token_table.py, fails: slot_of, says: "one entry per *block*" }
  - { file: block_table/mistakes/no_lookup.py, fails: slot_of, says: "Look it up in the table." }
  - { file: block_table/mistakes/offset_not_mod.py, fails: slot_of, says: "a block only has slots" }
  - { file: block_table/mistakes/one_number.py, fails: slot_of, says: "one value" }
  - { file: block_table/mistakes/gather_placeholder.py, fails: gather_kv, says: "still returns `...`" }
  - { file: block_table/mistakes/physical_order.py, fails: gather_kv, says: "Use the block table" }
  - { file: block_table/mistakes/no_reshape.py, fails: gather_kv, says: "still one row per block" }
  - { file: block_table/mistakes/not_cut.py, fails: gather_kv, says: "Cut to `length`" }
  - { file: block_table/mistakes/flattened.py, fails: gather_kv, says: "one long row" }
  - { file: block_table/mistakes/hardcoded_d.py, fails: gather_kv, says: "Read d from the pool" }
  - { file: block_table/mistakes/no_main.py, fails: round_trip, says: "is missing" }
```

### Attention over pages

```challenge
id: paged_attention
file: paged.py
minutes: 25
start: { from: block_table }
setup:
  - say: "Gathering works. Now use it: one new query token attends to everything this request has cached, keys and values, both stored in pages."
    type_file: paged_attention/stub.py
    pause: 2.4
  - say: "The lines at the bottom fill a value pool the same way, then compare your paged attention with PyTorch's dense attention over the original keys and values."
    pause: 1.2
tests: paged_attention/tests.py
solution: paged_attention/solution.py
alternatives:
  - paged_attention/alternatives/sdpa.py
  - paged_attention/alternatives/transpose.py
  - paged_attention/alternatives/math_sqrt.py
hints:
  nudge: Attention doesn't care where the keys lived. Get them into one tensor, in order, with the function you just wrote. Then it's the attention from lesson 1.1.
  concept: "`gather_kv(k_pool, block_table, length)` gives the `[length, d]` keys, and the same call on `v_pool` gives the values. `k @ q` is one score per cached token. Divide by √d, where d is `k.shape[-1]`, and `torch.softmax(..., dim=-1)` turns the scores into weights. `weights @ v` mixes the value rows."
  skeleton: |
    k = gather_kv(______, block_table, ______)
    v = gather_kv(______, block_table, length)
    weights = torch.softmax(k @ _ / k.shape[-1] ** ___, dim=-1)
  solution: |
    k = gather_kv(k_pool, block_table, length)
    v = gather_kv(v_pool, block_table, length)
    weights = torch.softmax(k @ q / k.shape[-1] ** 0.5, dim=-1)
on_solve: "Max abs diff 0. Gathering moves numbers without changing them, so paged attention is dense attention. Real kernels like vLLM's skip the copy: they read each block in place through the table while they compute, which adds fp16 rounding differences of about 1e-3. Same idea, fused."
mistakes:
  - { file: paged_attention/mistakes/placeholders.py, fails: returns_vector, says: "is still `...`" }
  - { file: paged_attention/mistakes/full_blocks.py, fails: matches_dense, says: "stale keys and values" }
  - { file: paged_attention/mistakes/no_scale.py, fails: matches_dense, says: "without the 1/√d scale" }
  - { file: paged_attention/mistakes/scale_d.py, fails: matches_dense, says: "not √d" }
  - { file: paged_attention/mistakes/v_from_k.py, fails: matches_dense, says: "mixes the *keys*" }
  - { file: paged_attention/mistakes/no_softmax.py, fails: matches_dense, says: "raw scores" }
  - { file: paged_attention/mistakes/no_main.py, fails: script_runs, says: "are missing" }
```

### Page your 8B model's cache

```replay
id: paged_8b
intro: Swapping your model's KV cache for a paged one and replaying 200 requests takes about 40 minutes on the GPU. Watch a sample run first so you know what healthy looks like.
log: replays/paged-8b.jsonl
speed: 40
recorded:
  hardware: RTX 3090 24 GB
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: Attention output max abs diff, paged vs dense
  flag_above: 0.01
  x_labels: [layers.0, layers.28]
  caption: Every layer agrees to fp16 rounding, and the greedy tokens are identical everywhere. A slot read from the wrong place would show up as a diff near 1 in that layer and every layer after it.
summary:
  - { label: Logits max abs diff, value: "0.0039" }
  - { label: KV reserved but empty, value: 82.9% → 1.1% }
  - { label: Requests that fit in 4 GiB, value: 8 → 46 }
run_yourself: |
  # on your GPU machine, from the course repo
  python paged_kv.py --ckpt runs/mn-8b --block-size 16 \
    --check-prompts 20 --replay requests_200.jsonl
```

### Record your metrics

```metrics
id: record
intro: One quality metric (does paging change the logits?) and one systems metric (how much KV memory still sits empty), from your own run.
```

## Hints

- vLLM's default block size is 16 tokens. Bigger blocks mean shorter tables but more waste in each request's last block.
- In a real model every layer has its own pools, shaped `[num_blocks, block_size, kv_heads, head_dim]`. One block table per request serves all of them: block 7 means slot 7 in every layer's pool.
- When a request finishes, its blocks go back on a free list. Any free block fits any request, so the memory never fragments.
- Two requests that start with the same prompt can point their tables at the same physical blocks. That's prefix caching, in lesson 5.4.

## Done when

- [ ] Your KV budget counts kv_heads, and paged waste beats contiguous {#kv_budget auto=challenge:kv_budget}
- [ ] Paged attention matches dense attention {#paged_attention auto=challenge:paged_attention}
- [ ] Your 8B model generates the same tokens through a paged cache on the GPU {#model_paged}
- [ ] Logit diff and KV waste recorded {#metrics_recorded auto=metrics}
