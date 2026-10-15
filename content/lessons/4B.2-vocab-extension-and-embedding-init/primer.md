---
gist: Adding 8,000 tokens means adding 8,000 rows to two matrices inside the model. What you put in those rows decides whether the model starts out nearly as good as before or worse than guessing, and the loss before any training tells you which.
minutes: 17
---

## What this lesson is for {#why}
> 4B.1 chose 8,000 new Mongolian tokens. This lesson gives the model a place to store each one, fills those places sensibly, and checks the result before any GPU time is spent training.

A tokenizer only produces ids. The model still has to know what each id means, and that knowledge lives in two big matrices. Llama 3.1 has a row in each for every one of its 128,256 tokens. Your 8,000 new tokens have no rows yet.

So this lesson does three things:

1. **Grow** both matrices by 8,000 rows without disturbing the old ones.
2. **Fill** each new row with a starting value.
3. **Check** the loss at step 0, before training, to see whether those starting values help or hurt.

4B.3 then trains the new rows on Mongolian text. A bad start here wastes that run.

## What you need from before {#before}
> An embedding table turns a token id into a vector by picking a row. The LM head turns the model's final vector back into one score per token, and softmax plus cross-entropy turn those scores into a loss.

You met all of these from makemore through 1.2:

- **Embedding**: a table with one row per token. Token id 7 means "take row 7". In Llama 3.1 each row is 4,096 numbers.
- **Hidden vector**: what the transformer produces at each position, also 4,096 numbers.
- **LM head**: a linear layer from 4,096 numbers to one score per token. Its weight is a `[vocab, 4096]` matrix, one row per token.
- **Logits**: those raw scores, one per token in the vocabulary.
- **Softmax**: turns logits into probabilities that add up to 1.
- **Cross-entropy loss**: −ln(probability of the token that really came next), averaged over positions. Lower is better.

Some models share one matrix for the embedding and the LM head ("tied weights"). Llama 3.1 8B doesn't. It has two separate matrices, so both need new rows.

## The two matrices and their shapes {#shapes}
> Both matrices are `[128256, 4096]` today and become `[136256, 4096]`. Only the number of rows changes.

```
embed_tokens.weight
  before  [128256, 4096]
  after   [136256, 4096]

lm_head.weight
  before  [128256, 4096]
  after   [136256, 4096]
```

At every position the model reads one embedding row (a lookup) and computes one logit per vocabulary row (a matrix product):

```
id 7  →  embed row 7      [4096]
  ...32 transformer layers...
hidden h                  [4096]
logits = h @ lm_head.T    [136256]
```

Every new row in `lm_head` adds one more logit at every position, even when the text uses no new token. Keep that in mind, because it's why a bad start hurts everything.

New parameters: 8,000 rows × 4,096 numbers × 2 matrices = **65.5M**. Training them in fp32 with Adam costs 16 bytes each (weight, gradient and two moments), about 1 GB. That's why 4B.3 trains only these rows in full and the rest of Llama through LoRA.

## Adding rows without moving old ones {#append}
> New tokens get the ids after the old ones, so new rows go at the end. Put them anywhere else and every old id reads the wrong row.

The extended tokenizer keeps every old token's id. The 8,000 new tokens get ids 128,256 to 136,255. So the old rows must stay where they are and the new rows go **after** them. That's joining along dimension 0, the row dimension:

```
old   [128256, 4096]
new   [  8000, 4096]
join along dim 0
      [136256, 4096]
```

The classic mistake is putting the new rows **first**. Then row 0 is a new token's row, and every old id points 8,000 rows too early. The pretrained model is scrambled, and the step-0 loss shows it immediately.

The browser challenge does this on a toy model whose matrices are `[102, 16]`: 102 base tokens, 16 numbers per row. With 20 new tokens both become `[122, 16]`.

## What should a new row start as? {#options}
> A new row needs starting values. The obvious choices are random numbers, zeros, the average of all rows, or the average of the rows of the pieces the token used to split into.

| Start | Idea | Problem |
|---|---|---|
| Random N(0, 1) | the default for a fresh `nn.Embedding` | rows far larger than trained ones |
| Small random | random but tiny | knows nothing about the token |
| Zeros | neutral | every new token identical |
| Mean of all rows | an "average token" | every new token identical |
| Mean of its pieces | start from what it meant before | the course's choice |

The last option uses something you already know. Before you added **байна** (is), Llama spelled it with existing tokens. In the toy vocab those are **ба**, **й** and **на**. The model already understands that sequence, so averaging those three rows gives the new token a starting point that means roughly the right thing.

## Why random rows hurt {#softmax}
> Softmax shares probability across every token. A new row with a large random logit steals probability from the right answer at every position.

Take three old tokens with logits `[2, 1, 0]`, where the first is the correct next token:

```
e^2 = 7.39, e^1 = 2.72, e^0 = 1
total = 11.11
p(correct) = 7.39 / 11.11 = 0.665
loss = −ln 0.665 = 0.41
```

Now add one new token. Its logit is its LM-head row times the hidden vector, so it depends on what you put in the row:

| New row gives logit | Total | p(correct) | Loss |
|---|---|---|---|
| (no new token) | 11.11 | 0.665 | 0.41 |
| 5 (random, large) | 159.5 | 0.046 | 3.07 |
| 1 (average of old) | 13.83 | 0.534 | 0.63 |

One random row turned a confident right answer into a 5% guess. With thousands of random rows, some of them will score high at almost every position. The text never mentions those tokens, but they still take probability away from the tokens that are right.

An averaged row can't do that. Its logit is the average of the logits of the rows it was made from, so it never beats the best of them.

## Averaging a token's pieces {#mean}
> For each new token, look up the rows of its base pieces and average them down the piece dimension, giving one row of the same width.

A tiny version with 2-number rows:

```
row of ба   [1.0, 0.0]
row of й    [0.0, 2.0]
row of на   [2.0, 1.0]

stacked     [3, 2]
mean down
the pieces  [1.0, 1.0]
```

Shapes at each step: the pieces' rows are `[pieces, width]`, and averaging across the pieces gives `[width]`, one new row. Stacking all 8,000 new rows gives `[8000, width]`, ready to append.

Mistakes the tests check for:

- **Averaging the wrong dimension.** Averaging across the width gives one number per piece instead of one row.
- **Summing instead of averaging.** A three-piece token starts about three times too long.
- **Using only the first piece.** "байна" would start out meaning just "ба".
- **Averaging every row** instead of the token's own pieces. All new tokens start out identical.

## Step-0 loss {#step0}
> Step-0 loss is the model's ordinary next-token loss on held-out Mongolian, measured before any training. It's a cheap check that the new rows didn't break anything.

To compute it, every token except the last predicts the one after it:

```
ids      [a, b, c, d]
inputs   [a, b, c]
targets  [b, c, d]
```

In the toy model there are no transformer layers. The logits are the input tokens' embedding rows times the LM head:

```
E rows   [n−1, 16]
H.T      [16, vocab]
logits   [n−1, vocab]
```

Then cross-entropy of those logits against the targets, averaged over positions.

Mistakes the tests check for: scoring each token against itself (no shift), predicting the token before, applying softmax before cross-entropy (it already does that), summing instead of averaging (a sum grows with the text's length), and using `E` as the head (Llama's head is its own matrix).

**A yardstick.** A model that knows nothing gives every token the same probability, 1/vocab. Its loss is ln(vocab): ln 122 ≈ 4.80 for the toy and ln 136,256 ≈ 11.82 for extended Llama. A start above that is worse than knowing nothing.

## Comparing across tokenizers {#per-char}
> Per-token losses from two tokenizers can't be compared, because their tokens cover different amounts of text. Convert to bits per character, as in 4A.4.

An extended token covers about twice as much text as a Llama token, so predicting it is harder and its loss is naturally higher. 4A.4 gave you the fix: divide by ln 2 to get bits, then by the characters each token covers.

| | nats per token | chars per token | bits per char |
|---|---|---|---|
| Llama 3.1 | 1.77 | 2.45 | 1.77 / 0.693 / 2.45 ≈ 1.04 |
| Extended, step 0 | 3.83 | 5.11 | 3.83 / 0.693 / 5.11 ≈ 1.08 |

So the extended model starts about 4% worse per character, with 8,000 rows that have never been trained. That's a healthy start. 4B.3 has to close that gap and go past it.

## What the toy shows {#toy}
> On the toy model, random rows start worse than a uniform guess and mean-of-pieces rows start best.

The browser challenge prints the step-0 loss for each start on the toy model (nats per token):

| Start | Loss |
|---|---|
| Random N(0, 1) | 6.39 |
| Uniform guess (ln 122) | 4.80 |
| Zeros | 4.08 |
| Mean of all rows | 3.89 |
| Mean of pieces | 3.23 |

The base model scores 2.03 nats per token, but on 128 tokens. The extended model needs only 77 tokens for the same text. Total loss is 2.03 × 127 ≈ 258 nats against 3.23 × 76 ≈ 245, so the extended model is already slightly ahead on the whole text.

## What you'll run {#run}
> On your GPU: resize Llama 3.1 8B's two matrices, fill the new rows from their Llama pieces, and measure step-0 loss on the 4A.3 held-out split. The recording in the lesson is illustrative.

The run takes a few minutes on a 12 GB card, with the transformer loaded in 4-bit and the two matrices in bf16. `resize_token_embeddings` adds the rows. Recent versions of Hugging Face transformers fill them with values sampled around the mean of the old rows. You then overwrite them with the mean of each token's pieces.

Healthy output looks like this (reference run, illustrative):

| Start | Step-0 loss |
|---|---|
| Random N(0, 1) | 15.9, above ln 136,256 = 11.82 |
| Random N(0, 0.02) | 5.02 |
| Mean of all rows (HF) | 4.41 |
| Mean of pieces | 3.83 (1.08 bits per char) |

If your loss is above ln(vocab), stop and look for misaligned rows, wrong pieces or an overflow. You record the step-0 loss (pass: under 5.0) and the new vocab size, 136,256.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Embedding (embed_tokens)**: the table that turns a token id into a row of numbers.
- **LM head (lm_head)**: the layer that scores every vocabulary token from the final hidden vector. One row per token.
- **Tied weights**: one matrix used as both embedding and LM head. Llama 3.1 8B doesn't tie them.
- **Hidden size**: the width of each row, 4,096 for Llama 3.1 8B and 16 in the toy.
- **Logit**: a raw score for one token, before softmax.
- **Softmax**: turns logits into probabilities that sum to 1.
- **Cross-entropy**: −ln(probability of the correct token), averaged over positions.
- **Resize**: grow a matrix to more rows, keeping the old rows in place.
- **Mean-of-pieces init**: starting a new token's row as the average of the rows of the base tokens it used to split into.
- **Step-0 loss**: the loss before any training. A check of the initialisation.
- **Uniform guess**: equal probability for every token. Its loss is ln(vocab size).
- **Nats / bits**: loss units. Nats use the natural log, and bits = nats / ln 2.
- **Bits per character**: loss divided by ln 2 and by the characters a token covers. Comparable across tokenizers.
- **new_ids.json**: the list of new token ids that 4B.3 trains.
