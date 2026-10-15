---
gist: Running many prompts in one forward pass is nearly free on a GPU, but prompts have different lengths, so you pad them on the left and give the model per-row positions and an attention mask that make each row behave exactly as if it ran alone.
minutes: 15
---

## What this lesson is for {#why}
> One decode step costs about the same for 1 prompt as for 16, so batching multiplies throughput. The hard part is making a padded batch give exactly the same answers as each prompt alone.

With the KV cache from 2.2, one decode step of your 30M model takes about 2 ms, and nearly all of it is overhead and reading weights, not arithmetic. Run 16 prompts through that same step and it barely gets slower, so you get 16 tokens instead of 1.

The catch is that a batch is a rectangle, a `[B, T]` tensor, and prompts have different lengths. This lesson builds the three pieces that make a ragged set of prompts into a correct batch:

1. **Left padding**, with a mask that records which tokens are real.
2. **Positions** counted from each row's first real token.
3. An **attention mask** that's causal and hides the padding.

The test is strict: every row must write exactly what its prompt writes alone. Serving systems (phase 5) live on batching, so this contract matters.

## What you need from before {#before}
> Shapes with a batch dimension, the causal mask from 1.1, position embeddings, and broadcasting from 0.0.

- **Batch shapes.** Your 1.x model takes ids `[B, T]` and returns logits `[B, T, V]`, one row of the batch per sequence.
- **Position embeddings.** Each token's input is its token embedding plus a position embedding. Position 0, 1, 2… each have their own vector.
- **The causal mask.** A `[T, T]` grid where query t (row) may look at key s (column) only if s ≤ t.
- **Broadcasting.** Sizes of 1 stretch to match. A `[B, 1, T]` tensor combined with a `[T, T]` tensor gives `[B, T, T]`.
- **Generation reads the last column.** In 2.1 the new token came from `logits[-1]`. With a batch it's `logits[:, -1]`, one row per sequence.

## Why batching is nearly free {#why-batch}
> At batch 1 the GPU spends a step reading every weight to make one token. A batch reuses that same read for every row.

Think of a web server that fetches a big config file from disk on every request. Serving 16 requests off one fetch is far cheaper than fetching it 16 times. A decode step is like that: reading the weights is the expensive part, and every row in the batch shares the read.

In the illustrative recording for your 30M model:

| Batch | Time per step | Tokens/s |
|---|---|---|
| 1 | 2.1 ms | 480 |
| 4 | 2.2 ms | 1,820 |
| 16 | 3.0 ms | 5,380 |
| 64 | 5.4 ms | 11,900 |

Batch 16 gives 11.2× the throughput of batch 1, for a step that's only about 40% slower. Lesson 2.5 explains exactly when this stops working.

## Padding into a rectangle {#padding}
> Shorter prompts are filled out with a pad id until every row is as long as the longest prompt, and a mask records which positions hold real tokens.

Three prompts of lengths 7, 5 and 3 become a `[3, 7]` batch. Here `P` marks a pad:

```
"the cat"  t h e _ c a t
"a dog"    P P a _ d o g
"the"      P P P P t h e
```

The mask has the same shape and is `True` on real tokens:

```
row 0  T T T T T T T
row 1  F F T T T T T
row 2  F F F F T T T
```

Record the mask from the positions you wrote, not by checking which ids equal the pad id. In the tiny model the pad id is 0, and so is the space character, the `_` above. A mask built from the ids would call those spaces padding.

## Why the pads go on the left {#left}
> Generation reads the last column. With left padding, every row's last column is a real token, and new tokens for every row go in the same next column.

Compare right padding for "a dog":

```
left:   P P a _ d o g  → next
right:  a _ d o g P P  → next
```

With right padding, the last column of this row is a pad. `logits[:, -1]` would give the prediction made after a pad, not after "g". Left padding keeps all rows aligned at the right edge, so one `torch.cat` of a new column extends every row correctly.

## Positions per row {#positions}
> A padded row's first real token must still be position 0, or the model sees a different sequence than the prompt alone.

If you number columns 0 to 6 for every row, "a dog"'s first real token gets position 2, not 0. The model has never seen "a dog" starting at position 2 in this prompt's solo run, so its outputs change.

Count positions over real tokens only. Here's the idea on one row, with the mask as 0s and 1s:

| | col 0 | col 1 | col 2 | col 3 |
|---|---|---|---|---|
| mask | 0 | 0 | 1 | 1 |
| running count | 0 | 0 | 1 | 2 |
| minus 1 | −1 | −1 | 0 | 1 |
| pads raised to 0 | 0 | 0 | 0 | 1 |

The real tokens get 0 and 1, exactly as if the row had no padding. The pads' positions don't matter, because no real token will ever attend to them, but they must be valid. Position −1 would crash the position-embedding lookup with `index out of range`.

The tests also recognise positions that start at 1 instead of 0.

## The attention mask, [B, T, T] {#keep}
> Query t may attend to key s only if s ≤ t (causal) and key s is a real token. Both conditions must hold, and each row of the batch gets its own grid.

For a 2-token prompt `a b` padded to 4 columns as `P P a b`, with rows as queries and columns as keys:

```
       keys: P  P  a  b
query P     .  .  .  .
query P     .  .  .  .
query a     .  .  ✓  .
query b     .  .  ✓  ✓
```

The causal grid allows the lower triangle. The key mask removes the two pad *columns*. What's left: real tokens see real tokens at or before them, and nothing else.

Shapes: the causal grid is `[T, T]` and the same for every row. The key flags are `[B, T]`. To broadcast them into `[B, T, T]`, the flags need a size-1 *query* dimension in the middle, `[B, 1, T]`, so the same column flags apply to every query row.

Mistakes the tests recognise:

| What you did | What goes wrong |
|---|---|
| Causal mask only | Real tokens attend to pads |
| Flags on the query axis | Hides pad rows, not pad columns |
| Padding mask only | Queries see the future |
| "or" instead of "and" | Queries see the future |
| One `[T, T]` mask | Ignores each row's padding |

## Rows with nothing to see {#empty-rows}
> A pad query at the start of a left-padded row has no allowed keys. With -inf its softmax would be NaN, so the model fills masked scores with the most negative float instead.

Look at the first two rows of the grid above: no ✓ at all. Softmax over a row that's all `-inf` divides 0 by 0 and gives NaN. In a one-layer model that NaN stays in the pad rows, which nobody reads. In a deep model the pad positions' NaNs flow into the next layer's values, and 0 × NaN is still NaN, so they can spread into real rows.

The batched tiny model avoids this by filling masked scores with `torch.finfo(dtype).min`, the most negative finite float. A row with nothing allowed gets a harmless uniform average. For real rows nothing changes, because e raised to that number is exactly 0 in float32. Hugging Face models use the same trick.

## The contract, and its cost {#contract}
> Batching may change the speed, never the answer. The price is wasted work on pads.

The final test runs each prompt alone and again inside the padded batch, and compares the real tokens' logits. They must match to about 10⁻⁴. In fp32 they match closely. In fp16 or bf16, tiny differences from a different order of additions are normal, so check correctness in fp32 first.

Padding costs compute. Four prompts of lengths 12, 40, 7 and 21, padded to 40, fill 160 slots with only 80 real tokens: half the prefill is wasted on pads. Sorting prompts by length before batching helps. Continuous batching, in 5.4, avoids the problem by changing the batch as sequences finish.

## On your GPU {#gpu}
> You'll run 64 prompts through your 30M model at batch sizes 1, 4, 16 and 64 with the KV cache on. The recording is illustrative sample data, not a real run.

The replay shows `generate.py --batch-sizes 1,4,16,64`. Healthy output looks like:

- Time per step barely moving up to batch 16 (about 2.1 → 3.0 ms), then rising by batch 64.
- Throughput going from 480 to about 5,380 tokens per second at batch 16, an 11.2× gain.
- All 64 prompts producing the same text batched as alone.
- KV cache growing with the batch, to about 0.10 GiB at batch 16.

The numbers are sample data, labelled as such in the app. You record the share of prompts whose batched output matches (at least 95%) and the throughput at batch 1 and batch 16. The gain is their ratio.

With a KV cache, batching works the same way. Each row's cache starts with pads, the mask keeps them hidden, and each step appends one column for every row. A row that finishes early keeps generating unless you track a `done` flag for it.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Batch (B)**: several sequences processed together in one tensor.
- **Padding / pad id**: filler tokens that make every row the same length.
- **Left padding**: pads go before the prompt, so every row ends on a real token.
- **Mask (padding mask)**: `[B, T]` bools, True on real tokens.
- **Positions (pos)**: `[B, T]` indexes into the position embedding, counted from each row's first real token.
- **Attention mask (keep)**: `[B, T, T]` bools. True where query t may attend to key s.
- **Query axis / key axis**: in `[B, T, T]`, the middle dimension is queries (rows), the last is keys (columns).
- **Causal mask**: allows key s for query t only if s ≤ t.
- **torch.finfo(dtype).min**: the most negative finite number a float type can hold.
- **NaN**: "not a number", the result of 0 ÷ 0. It spreads through any arithmetic it touches.
- **Throughput**: tokens produced per second across the whole batch.
- **Throughput gain**: tokens/s at batch 16 ÷ tokens/s at batch 1, this lesson's systems metric.
- **Batch match**: the share of prompts whose batched output equals their solo output, this lesson's quality metric.
- **Continuous batching**: adding and removing sequences from a running batch, covered in 5.4.
