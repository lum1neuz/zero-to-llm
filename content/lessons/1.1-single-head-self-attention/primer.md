---
gist: Attention lets every token look back at all the tokens before it and take a weighted average of what they carry, with the weights decided by how well each pair matches.
minutes: 18
---

## What this lesson is for {#why}
> Your bigram model only ever sees the previous character. Attention lets each position look at every earlier position and decide for itself which ones matter.

In 0.3 the bigram model predicted the next character from the current one alone. It can't know that a `q` three letters back makes a `u` likely, or that a sentence started with a name. Real language needs context.

**Self-attention** is the mechanism that gives a model context. Every token asks a question of all the tokens before it, scores how well each one answers, and takes a weighted mix of what they offer. A GPT is a stack of these, plus some simpler layers you add in 1.2.

In this lesson you write one **attention head** by hand, on tensors small enough to check with a pencil. Then a recording shows what one head does to a real training run on Shakespeare. Everything later in the course builds on this: the full block in 1.2, the 30M model in 1.4, and the KV cache in phase 2, which exists only because of attention.

## What you need from before {#before}
> Shapes, broadcasting, embeddings and the loss from phase 0. A quick refresher.

- **Shapes.** A tensor of shape `[4, 8, 32]` is 4 sequences, each 8 tokens long, each token a vector of 32 numbers. In this course that's written `[B, T, C]`: batch, time (position) and channels (features).
- **Embeddings.** A token id becomes a vector by looking up a row of a learned table, as in makemore.
- **`nn.Linear(in, out)`** turns a vector of `in` numbers into `out` numbers with a learned matrix. Its `weight` has shape `[out, in]`.
- **Softmax and the loss.** In 0.3 you turned counts into probabilities and scored predictions with the negative log likelihood. Both come back here, so the next sections refresh softmax properly.

## Dot products {#dot}
> Multiply two vectors position by position and add the results. The answer is large when the vectors point the same way.

```
a = [1, 2, 3]
b = [4, 0, -1]

a·b = 1·4 + 2·0 + 3·(-1)
    = 4 + 0 - 3
    = 1
```

A dot product measures how much two vectors agree. Two vectors that point the same way give a big positive number, opposite directions give a big negative one, and unrelated directions give something near 0. Attention uses exactly this as its "how well does this key match my query" score.

## Matrix multiplication is many dot products {#matmul}
> `A @ B` dots every row of A with every column of B. The inner sizes must match, and the outer sizes give the result's shape.

If `q` holds one row per token and `k` holds one row per token, you want the dot product of *every* query with *every* key. That's a grid: row t, column s is "query t against key s".

Three tokens, two features each:

```
q = [[1, 0],     k = [[1, 0],
     [0, 1],          [1, 1],
     [1, 1]]          [0, 1]]
```

| | key 0 | key 1 | key 2 |
|---|---|---|---|
| query 0 `[1,0]` | 1 | 1 | 0 |
| query 1 `[0,1]` | 0 | 1 | 1 |
| query 2 `[1,1]` | 1 | 2 | 1 |

A matrix multiply makes that grid in one operation, but only if the shapes line up. `A @ B` needs A's last size to equal B's second-to-last size: `[3, 2] @ [2, 3]` gives `[3, 3]`. Both `q` and `k` are `[3, 2]`, so `q @ k` fails ("cannot be multiplied (3x2 and 3x2)"). The keys have to be turned on their side, rows into columns, so that each key becomes a column. Turning a matrix on its side is a **transpose**.

In Java terms, the grid is a double loop: for every query, for every key, sum the products. The matmul runs all of those loops at once.

## Softmax {#softmax}
> softmax turns a row of any numbers into positive weights that add up to 1. Bigger numbers get exponentially bigger weights.

For each number x, take eˣ, then divide by the total:

```
scores   [2,    1,    0   ]
e^score  [7.39, 2.72, 1.00]
total    11.11
weights  [0.665, 0.245, 0.090]
```

Three facts you'll use:

| Scores | Weights | Why |
|---|---|---|
| `[0, 0, 0]` | `[0.33, 0.33, 0.33]` | equal scores, equal shares |
| `[0, 0, −∞]` | `[0.5, 0.5, 0]` | e^−∞ is exactly 0 |
| `[8, 0, −8]` | `[1.00, 0.00, 0.00]` | big gaps go nearly one-hot |

In attention, softmax runs along each **row** of the score grid, so each query's weights over the keys add up to 1. On a `[T, T]` grid that's the last dimension, `dim=-1`. Normalising down the columns instead (`dim=0`) is one of the mistakes the tests catch: the columns add up to 1 and the rows don't.

## Queries, keys and values {#qkv}
> Each token makes three vectors: a query (what am I looking for?), a key (what do I contain?) and a value (what do I pass on if chosen).

Think of a search box. Your query is matched against each page's title (the key), and the results you read are the pages' contents (the values). Attention is a *soft* search: instead of returning the single best match, it returns a blend of every page's content, weighted by how well its title matched.

For each token:

1. Score its query against every earlier token's key, with dot products.
2. Turn the scores into weights with softmax.
3. Output the weighted average of those tokens' values.

In the Learn deck, "sat" in "The cat sat" puts most of its weight on "cat": its query asks "who did this?" and the key of "cat" answers well. Nobody programs that. Each of the three vectors comes from its own learned `nn.Linear`, and training finds weights that make useful matches.

## Hiding the future {#mask}
> During training, every position predicts the next token at once, so position t must not see tokens after t. A triangular mask hides them.

A GPT trains on whole sequences. Position 0 predicts token 1, position 1 predicts token 2, and so on, all in parallel. If position 1 could attend to token 2, it could just copy its answer. Training loss would collapse and generation would be garbage, because when you generate, token 2 doesn't exist yet.

The fix is a lower-triangular grid of ones, which marks the cells each position *may* see:

```
         key 0  1  2  3
query 0  [   1  0  0  0 ]
query 1  [   1  1  0  0 ]
query 2  [   1  1  1  0 ]
query 3  [   1  1  1  1 ]
```

`torch.tril` builds this from a grid of ones. Every cell that's 0 here is the future. Its score is replaced with −∞ **before** softmax, so it gets exactly zero weight and the remaining weights still add up to 1.

Back to the three-token example, with the future masked (and the √d scaling skipped for now):

```
row 0  [1, -∞, -∞] → [1.00, 0,    0   ]
row 1  [0,  1, -∞] → [0.27, 0.73, 0   ]
row 2  [1,  2,  1] → [0.21, 0.58, 0.21]
```

Row 0 can only look at itself. Row 2 sees everything and leans towards key 1, its best match.

## Mask mistakes and their symptoms {#mask-mistakes}
> Each wrong mask leaves a recognisable fingerprint in the weights, and the tests read those fingerprints.

| Mistake | What the weights look like |
|---|---|
| no mask | non-zero weights above the diagonal |
| fill with 0 instead of −∞ | same: e⁰ = 1, a real share |
| `triu` instead of `tril` | zeros *below* the diagonal |
| hide where tril is 1 | the last row is NaN (0 / 0) |
| mask after softmax | rows add up to less than 1 |
| softmax over `dim=0` | columns add up to 1, rows don't |

The NaN case is worth understanding. If a row has every score set to −∞, softmax computes 0 / 0. That happens when the mask hides each token's past and keeps its future: the last token has no future, so nothing is left.

## Why divide by √d {#scale}
> Dot products of d-number vectors grow like √d. Dividing by √d keeps the scores around ±1, where softmax stays soft and trainable.

If q and k hold 64 random numbers each, with variance 1, then q·k is a sum of 64 products, each with variance 1. Variances of independent terms add, so q·k has variance 64 and a typical size of √64 = 8.

```
raw scores   [8, 0, -8]
softmax      [1.00, 0.00, 0.00]

÷ √64 = 8    [1, 0, -1]
softmax      [0.665, 0.245, 0.090]
```

Unscaled, every row is nearly one-hot from the very first step. A one-hot softmax barely changes when its inputs change, so its gradients are tiny and the head learns slowly. The scale is `1 / √d` where d is the head's feature size, the last dimension of q.

The tests check for the near misses: dividing by d instead of √d (rows flatten out), multiplying instead of dividing, dividing by √T (the number of tokens, which has nothing to do with it), or dividing by a fixed 2 that only happens to equal √d when d = 4.

## Shapes through one head {#shapes}
> x is [B, T, C]. The head turns it into k, q and v of [B, T, hs], a weight grid of [B, T, T], and an output of [B, T, hs].

Step 2's script uses B = 4, T = 8, C = 32 and a head size of 16:

| Tensor | Shape | Meaning |
|---|---|---|
| x | `[4, 8, 32]` | 4 sequences of 8 token vectors |
| k, q, v | `[4, 8, 16]` | one key, query, value per token |
| scores, weights | `[4, 8, 8]` | every query × every key |
| output | `[4, 8, 16]` | a blended value per token |

The batch dimension rides along untouched: each sequence gets its own `[8, 8]` grid. That's why the keys must swap only their **last two** dimensions. `k.T` reverses *every* dimension, which works on a single `[T, d]` matrix but turns a batch of `[4, 8, 16]` into `[16, 8, 4]`, and the matmul fails. Build the mask from the input's own T, too: a mask fixed at the demo's 5 × 5 breaks the moment T changes, which it does every time a model generates one more token.

Two output mistakes are easy to spot from the shape alone. Returning the weights gives `[B, T, T]`, and mixing x instead of v gives `[B, T, C]`.

## What you'll build {#build}
> Step 1 writes the masked, scaled weights on a 5 × 4 example. Step 2 wraps them in a Head with learned projections and times it at T = 256.

**Step 1, `causal_weights`.** Five tokens, each with a 4-number query and key. You score, mask and softmax. The tests try a batch, a different T and a larger d, and tell you which fingerprint from the tables above your weights match.

**Step 2, `Head`.** Three `nn.Linear` layers without bias make k, q and v from x, and your step 1 function turns q and k into weights. A test changes the last tokens of a sequence and checks that the earlier outputs don't move: the plainest possible check that nothing sees the future. The script then times the head on a 256-token sequence. That time is your systems metric; on the reference machine it's about 3.3 ms in the browser. Try T = 512 afterwards: the score grid is T × T, so doubling T makes it four times bigger.

## The Shakespeare run {#run}
> On your laptop, one head between the embeddings and the output layer brings Tiny Shakespeare's validation loss from about 2.49 to about 2.40.

The recording shows a character-level model with one head (32 features, 8 characters of context) trained for 5,000 steps on a CPU, about a minute and a half. It's **illustrative**: sample data written for the course, not a real recording.

Healthy looks like this. The loss starts near 4.2, which is ln(65): guessing uniformly among 65 characters. It falls fast, then slowly. Around step 3,000 the validation loss drops below 2.49, the bigram table's best, and it ends near 2.40. The samples are still gibberish, but they spell better, because each character now sees up to 8 characters back. Validation loss is the same negative log likelihood as in 0.3, measured in **nats** (natural-log units), on text the model never trained on.

Your metrics are your own val loss (pass under 2.47) and the T = 256 time from step 2.

## Words you'll see {#glossary}
> The terms in this lesson's cards, challenges and metrics.

- **Attention head**: one set of query, key and value projections plus the weighted average they produce.
- **Self-attention**: attention where the queries, keys and values all come from the same sequence.
- **Query, key, value (q, k, v)**: per-token vectors for "what I'm looking for", "what I contain" and "what I pass on".
- **Score**: the dot product of a query with a key, before softmax.
- **Weights (wei)**: the scores after masking and softmax. Each row adds up to 1.
- **Causal mask**: the triangular grid that hides later positions. `tril` keeps the lower triangle.
- **−∞ (`float("-inf")`)**: the score that softmax turns into exactly 0.
- **Scaled dot-product attention**: softmax(q·kᵀ / √d) times v.
- **d, head_size (hs)**: the number of features in one head's q, k and v.
- **B, T, C**: batch size, sequence length (time) and channels (features per token).
- **Transpose**: swapping two dimensions of a tensor; here the last two of k.
- **`bias=False`**: a linear layer with no added constant, just the matrix.
- **Nats**: the unit of a loss computed with the natural logarithm.
- **Validation loss**: the loss on held-out text, which tells you whether the model generalises.
