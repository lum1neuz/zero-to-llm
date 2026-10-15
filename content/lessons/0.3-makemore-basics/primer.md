---
gist: Makemore's first model guesses each letter of a name from the letter before it. You build it twice, once by counting letter pairs and once as a tiny neural network trained by gradient descent, and the two land in the same place.
minutes: 18
---

## What this lesson is for {#why}
> You build makemore's first real model, a bigram, and then rebuild it as a neural network, which is the form every later model in the course takes.

Makemore reads real names and invents new ones that sound similar. This lesson's version looks at one letter and predicts the next. It's called a **bigram** model, because it only ever considers pairs.

You build it two ways:

1. **By counting.** Count how often each letter follows each other letter, turn the counts into probabilities, and generate names by rolling weighted dice. Then score the model with the loss from 0.1.
2. **As a neural network.** Replace the count table with a table of weights, and let gradient descent, the training loop from 0.2, find good weights.

Both end up at about the same loss. So why bother with the second? Because counting stops working once the model looks at more than one letter, and the neural version keeps working all the way up to a transformer. The rest of the course grows this second version.

## What you need from before {#before}
> From 0.0: stoi, zip and keepdim. From 0.1: the loss. From 0.2: gradients and the update step.

| From | Idea | Used here for |
|---|---|---|
| 0.0 | `stoi` / `itos` map characters to 0–26 and back; "." is 0 | rows and columns of the table |
| 0.0 | `zip(chs, chs[1:])` gives every neighbouring pair | walking the bigrams of a name |
| 0.0 | `N / N.sum(1, keepdim=True)` makes each row sum to 1 | counts to probabilities |
| 0.0 | `P[xs, ys]` picks one cell per (x, y) pair | the loss with tensors |
| 0.1 | loss = average of −ln p over the right answers; uniform model = ln 27 ≈ 3.30 | scoring the model |
| 0.2 | `loss.backward()` fills `.grad`; step each weight against its grad | training the network |

## Counting pairs {#counting}
> Wrap each name in "." markers, walk its neighbouring pairs, and add one to the cell [first letter, next letter] of a 27×27 table.

Take a tiny vocabulary of three symbols, ".", "a" and "b", and three names: `ab`, `ba` and `aab`. Wrapped in markers they give ten pairs:

```
.ab.   →  .a  ab  b.
.ba.   →  .b  ba  a.
.aab.  →  .a  aa  ab  b.
```

Counting them into a table N, with the row as the first character and the column as the next:

```
           next:  .   a   b
row "."        [  0,  2,  1 ]
row "a"        [  1,  1,  2 ]
row "b"        [  2,  1,  0 ]
```

Row "." says names start with "a" twice and "b" once. Column "." says how often each letter ends a name. With the full alphabet the table is 27 × 27, and the lesson's 81 names fill it with 542 pairs: one per letter plus the final ".".

The tests catch three classic slips:

- **No markers.** Then row "." is empty, and the model never learns how names start or when they stop.
- **`=` instead of `+=`.** Every cell ends up 0 or 1, because each new pair overwrites the count instead of adding to it.
- **Row and column swapped.** The table comes out transposed: row i should be "everything that follows i".

## Counts to probabilities, and smoothing {#probs}
> Divide each row by its total to get P, the probabilities of each next letter. Add 1 to every count first, so nothing is impossible.

Dividing each row by its own total turns counts into a probability distribution per current letter. That's 0.0's `normalize_rows`, with `keepdim=True` so each row meets its own total.

There's one problem. Row "b" above has a 0: in the training names, "b" was never followed by "b". Unnormalised, that row gives `P(b | b) = 0`, and 0.1 showed that −ln 0 is infinite. A single test name containing "bb" would make the loss infinite.

The fix is **smoothing**: add 1 to every count before dividing.

| | . | a | b | total |
|---|---|---|---|---|
| row "b" counts | 2 | 1 | 0 | 3 |
| plus 1 | 3 | 2 | 1 | 6 |
| probabilities | 0.500 | 0.333 | 0.167 | 1 |

Now "bb" is unlikely but possible. Adding 1 matters more when data is scarce: with 81 names, many rows are mostly smoothing, which is one reason the samples look odd.

The `P(b | a)` notation in the cards reads "the probability of b, given that the current letter is a". It's the value in row "a", column "b".

## Generating names: weighted dice {#sampling}
> Start at ".", draw the next letter at random in proportion to that row's probabilities, move to the drawn letter's row, and stop when you draw "." again.

Think of each row of P as a raffle where each letter holds tickets in proportion to its probability. Starting from ".":

1. Draw from row ".". Say you get "e".
2. Draw from row "e". Say "m".
3. Keep going until a draw comes up ".", then stop. The letters drawn so far are the name.

PyTorch's `torch.multinomial(probs, num_samples=1, ...)` is the raffle: it returns an index chosen at random in proportion to the probabilities. Three details the tests check:

- **Draw, don't take the maximum.** Always taking the most likely letter (`argmax`) produces the same name every time. Generating means sampling.
- **Use the generator.** `torch.Generator().manual_seed(2147483647)` is like `new java.util.Random(seed)`: the same seed gives the same draws. Pass it to every draw, or your names change on every run.
- **Turn the result into a plain int.** The draw returns a one-element tensor. Dictionary lookups like `itos[ix]` need a Python int, and `.item()` converts it.

Expect gibberish. A bigram only remembers one letter, so it can't know that "xq" is unusual three letters into a name. Even with all 32,033 names the samples only get a little more name-like.

## Scoring the model {#scoring}
> Average −ln P over every pair in the data. Counting gets about 2.47 on the lesson's names, against the uniform model's 3.30.

The loss is exactly 0.1's: for every pair, take ln of the probability P gave to the letter that actually came next, average, and flip the sign. Worked example from the cards, for the name "ab":

| Pair | Probability | −ln |
|---|---|---|
| "." → "a" | 0.2 | 1.609 |
| "a" → "b" | 0.05 | 2.996 |
| "b" → "." | 0.5 | 0.693 |
| average | | **1.766** |

The unlikely middle pair contributes most of the loss. On the lesson's 81 names, the smoothed count table scores about 2.47 nats, 0.83 below the uniform 3.30. That gap is what counting pairs bought.

Mistakes the tests look for: reading P[next, current] instead of P[current, next], dividing by 27 (the vocabulary size) instead of the number of pairs, summing instead of averaging, and averaging the probabilities instead of their logs.

## Why a neural network? {#why-nn}
> A count table needs one row per possible context. That's 27 rows for one letter of context, but 19,683 for three letters, and most of them would be empty.

Counting works for one letter of context. To look three letters back you'd need a row for every three-letter combination: 27³ = 19,683 rows of 27 counts each. Most combinations never appear in 32,033 names, so most rows would be pure smoothing. Each extra letter multiplies the table by 27.

A neural network replaces the table with a function that has a fixed number of weights and *learns* them from data. For this lesson it is deliberately as simple as possible, so you can compare it with counting: a single 27 × 27 matrix of weights, W. The interesting part is not the model but the recipe, which stays the same from here to transformers:

1. Turn the input into numbers.
2. Multiply by weights to get a score for every possible next letter.
3. Turn the scores into probabilities.
4. Compute the loss, backpropagate, and step the weights.

## One-hot encoding: a multiply that picks a row {#one-hot}
> A one-hot vector is all zeros except a single 1. Multiplying it by a matrix picks out one row of that matrix.

Weights can only multiply numbers, so each input letter becomes a vector of 27 numbers: 1 at its own index and 0 everywhere else. "a" (index 1) becomes:

```
[0, 1, 0, 0, …, 0]   (27 numbers)
```

`F.one_hot(xs, num_classes=27)` does this for a whole tensor of indices. It returns integers; `.float()` turns them into floats for the multiplication.

Now multiply by W. With a 3-symbol example:

```
x = [0, 0, 1]

W = [[0, 1, 2],
     [3, 4, 5],
     [6, 7, 8]]

x @ W = 0·row0 + 0·row1 + 1·row2
      = [6, 7, 8]
```

The one-hot vector just selects row 2. So in this model, W is a lookup table, exactly like the count table: row i holds the scores for what follows letter i. The difference is that W is learned.

Shapes for the lesson's 542 training pairs:

| Tensor | Shape | What it is |
|---|---|---|
| `xs` | `[542]` | index of each current letter |
| `ys` | `[542]` | index of each next letter |
| one-hot `xenc` | `[542, 27]` | one row per input |
| `W` | `[27, 27]` | the weights |
| `logits = xenc @ W` | `[542, 27]` | 27 scores per input |

## Softmax and the loss, with tensors {#softmax}
> Read the scores, called logits, as log-counts: exponentiate to get positive counts, then divide each row by its total. Then pick each row's probability of its right answer.

W can hold any numbers, positive or negative, so the scores (**logits**) can't be probabilities directly. The trick is to read them as logarithms of counts and undo the log with `exp`:

```
logits     [ 2.0,   1.0,   0.0 ]
exp        [ 7.389, 2.718, 1.0 ]
sum        11.107
probs      [ 0.665, 0.245, 0.090 ]
```

`exp` makes every value positive and keeps bigger scores bigger. Dividing by the row total makes the row sum to 1. Those two steps together are called **softmax**.

Then the loss. For row i you want the probability at column `ys[i]`, the right answer. That's 0.0's `probs_of` trick: index with `torch.arange(n)` (the numbers 0 to n−1) for the rows and `ys` for the columns, which pairs them up and gives one probability per row, shape `[542]`. Take −ln, average, and you have the same loss as the count model.

Mistakes the tests check:

| Mistake | What you see |
|---|---|
| No `exp` | some "probabilities" are negative |
| Normalising columns (`dim=0`) or the whole table | rows don't sum to 1 |
| `probs[:, ys]` | a 542 × 542 table: every row scored on every target |
| Using the script's `n` inside the loss | breaks when the function gets a different number of rows |

## Training: gradient descent on W {#training}
> Each step: compute the loss, clear the old gradients, backpropagate, and move W a small step against its gradient. 100 steps take the loss from about 3.75 to 2.08.

This is 0.2's training loop, with PyTorch's autograd doing the backward pass:

1. **Forward**: one-hot, logits, softmax, loss.
2. **Clear** the old gradient by setting `W.grad = None`. Otherwise gradients add up across steps, as in 0.2.
3. **Backward**: `loss.backward()` fills `W.grad`, same shape as W.
4. **Update**: change every weight by `−learning_rate × its grad`.

Two details for the update:

- **Step against the gradient.** The gradient points uphill, towards a higher loss. Adding it makes the loss go up, and it can blow up to `nan` ("not a number").
- **Don't let autograd record the update.** W was created with `requires_grad=True`, so PyTorch refuses an in-place change like `W -= …` ("a leaf Variable that requires grad is being used in an in-place operation"). Change the numbers through `W.data`, or inside a `with torch.no_grad():` block.

The learning rate here is 50, far larger than usual. It works because this model is so simple. Starting from random W, the loss goes from about 3.75 to 2.08 in 100 steps.

Why 2.08 and not 2.47? The count model added 1 to every count. Without smoothing, counting gets 2.06 on these names, and gradient descent is closing in on exactly that: **W is learning the log of the count table**. Same model, found two ways.

## On your machine: all the names {#full-run}
> Run both versions on all 32,033 names, train on 90% and measure the loss on the 10% the model never saw. Both land near 2.45.

The browser uses 81 names. On your machine, download `names.txt` (the link is in the lesson's Hints) and use them all: 228,146 pairs.

Measure the loss on names the model was **not** trained on. A model can do well on data it has memorised and badly on new data, so the honest score is on held-out data. Shuffle the names with a fixed seed, train (or count) on the first 90%, and compute the loss on the last 10%. That's the **validation loss**.

| Metric | What to record | Reference | Pass |
|---|---|---|---|
| Validation loss | loss on the held-out 10% | about 2.45 | at most 2.5 |
| Training time | seconds for 100 steps of the neural bigram on all pairs | about 1.5 s on a recent GPU (an estimate) | at most 60 s |

A bigram can't do much better than about 2.45, because it only ever sees one letter. Bengio et al.'s model, makemore part 2, looks at three letters through a small table of learned letter vectors and gets near 2.2. Lesson 0.4 trains that model.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Bigram**: a pair of neighbouring characters, and the model that predicts each character from the one before it.
- **Count table (N)**: the 27 × 27 table of how often each character followed each other one.
- **P**: the table of probabilities, each row of N divided by its total.
- **P(b | a)**: the probability of "b" given that the current character is "a": row "a", column "b".
- **Smoothing**: adding 1 to every count so no pair has probability 0.
- **Sampling**: drawing at random in proportion to the probabilities.
- **Multinomial**: PyTorch's weighted draw, `torch.multinomial`.
- **Generator / seed**: a random-number source and the starting number that makes its draws repeatable.
- **`.item()`**: turns a one-element tensor into a plain Python number.
- **Negative log-likelihood (NLL)**: the average of −ln p over the right answers. The loss.
- **One-hot**: a vector of zeros with a single 1 at an item's index.
- **Weights (W)**: the numbers a network learns. Here a 27 × 27 matrix.
- **Logits**: the raw scores before softmax, read as log-counts.
- **Softmax**: exponentiate, then divide by the row total. Turns logits into probabilities.
- **Gradient descent**: repeatedly stepping the weights against their gradient to lower the loss.
- **Learning rate**: the size of each step. 50 in this lesson.
- **`requires_grad` / leaf**: a tensor PyTorch tracks for gradients; a leaf is one you created directly, like W.
- **`torch.no_grad()` / `.data`**: two ways to change weights without autograd recording it.
- **`nan`**: "not a number", what a loss becomes when training blows up.
- **Validation loss**: the loss on held-out data the model never trained on.
- **Training time**: this lesson's systems metric, seconds for 100 training steps on all pairs.
