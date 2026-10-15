---
gist: You build makemore's first model, which guesses each letter of a name from the letter before it. You build it twice, once by counting and once as a tiny neural network that learns, and both end up in the same place.
minutes: 25
---

## Start here {#why}
> This is the counting model from W1, but for letters in names. You build it by counting, then again as a neural network, the form every later model uses.

In W1, a phone keyboard guessed the next word by counting what usually came next. Makemore does the same thing with letters. It reads real names and learns which letter tends to follow which. Then it can make up new names.

This first version only looks at **one letter** to guess the next. A pair of neighbouring letters is called a **bigram**, so this is a **bigram model**.

You build it two ways:

1. **By counting.** Count how often each letter follows each other letter. Turn the counts into chances. Use the chances to make up names, and score the model with the loss from 0.1.
2. **As a neural network.** Replace the table of counts with a table of adjustable numbers, and let training (from 0.2) find good values.

Both end up with almost the same score. So why do it twice? Because counting stops working when the model needs to look at more than one letter. The second way keeps working, all the way up to the big models later in the course.

## What you need from before {#before}
> You need the letter numbers and pairs from 0.0, the loss from 0.1, and the training step from 0.2.

| Idea | Reminder | From |
|---|---|---|
| Letter numbers | `stoi` turns a letter into 0–26, and `itos` turns it back. "." is 0. | 0.0 |
| Pairs | `zip(chs, chs[1:])` walks every pair of neighbours | 0.0 |
| Rows that sum to 1 | dividing each row by its own total, with `keepdim=True` | 0.0 |
| Picking cells | `P[xs, ys]` picks one cell for each (x, y) pair | 0.0 |
| The loss | the average of −ln of the chance given to each right answer | 0.1 |
| The baseline | a model that knows nothing scores about 3.30 | 0.1 |
| Training | `loss.backward()` fills each `.grad`, then step each knob against its grad | 0.2 |

## Counting pairs {#counting}
> Add a "." to the start and end of every name, walk through the pairs of neighbours, and add one to the right cell of a table for each pair.

Let's shrink the problem so you can check everything by hand. Use only three symbols, ".", "a" and "b", and three tiny names: `ab`, `ba` and `aab`.

First, add a "." at both ends of each name. The front "." says "a name starts here" and the back one says "the name stops here". Then list each pair of neighbours:

```
.ab.   →  .a  ab  b.
.ba.   →  .b  ba  a.
.aab.  →  .a  aa  ab  b.
```

That's ten pairs. Now count them into a table. The **row** is the first letter of the pair, and the **column** is the letter that came next:

```
           next:  .   a   b
row "."        [  0,  2,  1 ]
row "a"        [  1,  1,  2 ]
row "b"        [  2,  1,  0 ]
```

Check one cell: row "a", column "b" is 2, because "ab" appears twice (in `.ab.` and `.aab.`).

Each row tells a story. Row "." says names started with "a" twice and with "b" once. Column "." says how often each letter ended a name.

With all 26 letters plus ".", the table is 27 by 27. In the lesson, 81 names give 542 pairs: one per letter, plus the final ".".

The tests catch three common slips:

| Slip | What goes wrong |
|---|---|
| Forgetting the "." markers | Row "." stays empty, so the model never learns how names start or stop. |
| Using `=` instead of `+=` | Every cell ends up 0 or 1, because each pair overwrites the count instead of adding to it. |
| Swapping row and column | The table comes out flipped. Row "a" should hold what comes *after* "a". |

## From counts to chances {#probs}
> Divide each row by its total to turn counts into chances. Add 1 to every count first, so nothing is ever impossible.

Take row "a" from the table: 1, 1 and 2, which adds up to 4. Divide each count by 4:

| | . | a | b | total |
|---|---|---|---|---|
| counts after "a" | 1 | 1 | 2 | 4 |
| chances | 0.25 | 0.25 | 0.50 | 1 |

So after "a", the model says: 25% stop, 25% another "a", 50% "b". Do this for every row and you get a table of chances, called **P**. This is 0.0's `normalize_rows`.

The cards write a chance like **P(b | a)**. Read it as "the chance of b, given that the current letter is a". It's the number in row "a", column "b": 0.50 here.

**One problem: zeros.** Row "b" has a 0: in these names, "b" was never followed by another "b". So the chance of "bb" would be 0. In 0.1 you saw that a chance of 0 gives infinite surprise. One test name with "bb" would make the loss infinite.

The fix is called **smoothing**: add 1 to every count before dividing.

| | . | a | b | total |
|---|---|---|---|---|
| counts after "b" | 2 | 1 | 0 | 3 |
| plus 1 | 3 | 2 | 1 | 6 |
| chances | 0.500 | 0.333 | 0.167 | 1 |

Now "bb" is unlikely, but possible. With only 81 names, many rows are mostly made of these extra 1s, which is one reason the made-up names look odd.

## Making up names {#sampling}
> Start at ".", pick the next letter at random using that row's chances, move to the picked letter's row, and stop when you pick "." again.

Think of each row as a raffle. Each letter holds tickets in proportion to its chance. A letter with a 50% chance holds half the tickets.

To make up a name:

1. Draw a ticket from row ".". Say it's "e".
2. Draw from row "e". Say it's "m".
3. Keep going until you draw ".", then stop.

The letters you drew are the new name.

In PyTorch the raffle is `torch.multinomial`. You give it a row of chances, and it hands back the number of one letter, picked at random according to those chances.

Three things the tests check:

- **Draw, don't just take the favourite.** If you always took the most likely letter (that's called `argmax`), you'd get the same name every time. Making names means drawing at random.
- **Use the seeded generator.** A **generator** is a source of random numbers. Giving it a starting number, called a **seed**, makes the draws repeatable: the same seed always gives the same names. The lesson passes one in as `g`. Use it for every draw.
- **Turn the result into a plain number.** The draw comes back as a tensor holding one number. Looking up `itos[ix]` needs a plain Python number, and `.item()` gives you one.

Expect odd names. The model only remembers one letter, so it can't tell that a name is getting too long or strange. Even with all 32,033 names, the results only look a little more like names.

## Scoring the model {#scoring}
> Use 0.1's loss: average the surprise over every pair in the names. Counting scores about 2.47 on the lesson's names, much better than 3.30.

For every pair in the names, look up the chance P gave to the letter that really came next. Take −ln of it (the surprise), and average.

A worked example from the cards, for the name "ab":

| Pair | Chance | Surprise, −ln |
|---|---|---|
| "." then "a" | 0.2 | 1.609 |
| "a" then "b" | 0.05 | 2.996 |
| "b" then "." | 0.5 | 0.693 |
| average | | **1.766** |

The unlikely middle pair adds the most surprise.

On the lesson's 81 names, the smoothed table scores about **2.47**. A model that knows nothing scored 3.30 in 0.1. So counting pairs made the model 0.83 better.

Common slips the tests catch: looking up the cell the wrong way round (next letter, then current letter), dividing by 27 instead of by the number of pairs, adding up instead of averaging, and averaging the chances instead of their surprises.

## Why a neural network? {#why-nn}
> A counting table needs one row for every possible context. With one letter that's 27 rows; with three letters it's 19,683, and most would be empty.

Counting is fine when the model looks back one letter: 27 rows. But to look back three letters, you'd need one row for every three-letter combination. That's 27 × 27 × 27 = 19,683 rows. Most combinations never appear in 32,033 names, so most rows would be nearly empty.

A **neural network** solves this differently. It has a fixed set of adjustable numbers, called **weights**, and it learns good values for them from the data. For this lesson we keep it as small as possible, so you can compare it with counting: one 27 by 27 table of weights, called **W**.

The recipe is what matters, because it stays the same right up to the big models:

1. Turn the input letter into numbers.
2. Multiply by the weights to get a score for every possible next letter.
3. Turn the scores into chances.
4. Work out the loss, run the backward pass, and adjust the weights.

## One-hot: a multiply that picks a row {#one-hot}
> A one-hot list is all zeros except for a single 1. Multiplying it by a table picks out one row of the table.

Weights can only be multiplied by numbers. So each input letter becomes a list of 27 numbers: a 1 in its own position and 0 everywhere else. This is called **one-hot**.

The letter "a" is number 1, so it becomes:

```
[0, 1, 0, 0, …, 0]   (27 numbers)
```

`F.one_hot(xs, num_classes=27)` does this for many letters at once. It makes whole numbers, and `.float()` turns them into decimals for multiplying.

Now multiply a one-hot list by a table, using the `@` grid multiplication from 0.1. With three symbols:

```
x = [0, 0, 1]

W = [[0, 1, 2],
     [3, 4, 5],
     [6, 7, 8]]
```

Each answer number is the one-hot list times one column of W, multiplied in pairs and added up. The 0s wipe out everything except row 2:

| Column | Working | Result |
|---|---|---|
| 0 | 0·0 + 0·3 + 1·6 | 6 |
| 1 | 0·1 + 0·4 + 1·7 | 7 |
| 2 | 0·2 + 0·5 + 1·8 | 8 |

So `x @ W` is just row 2 of W: [6, 7, 8]. In this model, W works like the counting table: row i holds the scores for what comes after letter i. The difference is that W is learned, not counted.

The shapes for the lesson's 542 pairs:

| Name | Shape | What it holds |
|---|---|---|
| `xs` | `[542]` | the number of each current letter |
| `ys` | `[542]` | the number of each next letter |
| `xenc` | `[542, 27]` | one one-hot row per current letter |
| `W` | `[27, 27]` | the weights |
| `logits` | `[542, 27]` | 27 scores for each current letter |

## Scores to chances: softmax {#softmax}
> Treat the scores as "counts, but logged". Undo the log with exp, which makes them positive, then divide each row by its total. This is called softmax.

The weights can be any numbers, positive or negative. So the scores, called **logits**, can't be used as chances directly.

The trick is to read each score as the log of a count. W4 showed that `exp` undoes a log: it raises e (about 2.718) to that power. That gives positive "counts", which you turn into chances by dividing by the row total, just like before.

A worked example with three scores:

| Step | Values |
|---|---|
| scores (logits) | 2.0, 1.0, 0.0 |
| exp of each | 7.389, 2.718, 1.0 |
| row total | 11.107 |
| divide by the total | 0.665, 0.245, 0.090 |

The chances add up to 1, and the biggest score still gets the biggest chance. These two steps together, exp and then divide by the total, are called **softmax**.

**The loss, with tensors.** For row i, you want the chance at column `ys[i]`, the right answer. That's 0.0's picking trick. Pair the row numbers 0, 1, 2, … (from `torch.arange(n)`) with `ys`, and you get one chance per row: 542 numbers. Then take −ln and average, exactly as before.

Mistakes the tests check:

| Mistake | What you see |
|---|---|
| Skipping exp | Some "chances" come out negative. |
| Dividing each column, or the whole table, instead of each row | The rows don't add up to 1. |
| Writing `probs[:, ys]` | A 542 by 542 table: every row scored against every answer. |
| Using the script's `n` inside the loss function | It breaks when the function is given a different number of rows. |

## Training the network {#training}
> Each step: work out the loss, clear the old grads, run the backward pass, and move W a small step against its grad. In 100 steps the loss falls from about 3.75 to 2.08.

This is 0.2's training loop, with PyTorch doing the backward pass for you:

1. **Forward**: one-hot, scores, softmax, loss.
2. **Clear** the old grad with `W.grad = None`. Otherwise grads add up from step to step, as in 0.2.
3. **Backward**: `loss.backward()` fills `W.grad`, which has the same shape as W.
4. **Update**: change each weight by minus the learning rate times its grad.

Two details for the update:

- **Step against the grad.** The grad points in the direction that makes the loss *worse*. If you add it instead of taking it away, the loss goes up, and can blow up to `nan`, which means "not a number".
- **Don't let PyTorch track the update.** W was made with `requires_grad=True`, so PyTorch refuses a direct change like `W -= ...`. It says "a leaf Variable that requires grad is being used in an in-place operation". Change the numbers through `W.data`, or inside a `with torch.no_grad():` block.

The learning rate in this lesson is 50, which is much bigger than usual. It works because this model is so simple. From random weights, the loss falls from about 3.75 to 2.08 in 100 steps.

Why 2.08 and not 2.47? The counting model added 1 to every count. Without that extra 1, counting scores 2.06 on these names, and training is closing in on exactly that. **W is learning the log of the counting table.** It's the same model, found two different ways.

## On your own computer {#full-run}
> Run both versions on all 32,033 names. Train on 90% of them and score on the other 10%. Both land near 2.45.

The browser uses 81 names. On your computer, download `names.txt` (the link is in the lesson's Hints) and use all of them: 228,146 pairs.

This time, score the model on names it **never saw** during training. A model can do well on names it has memorised and badly on new ones. Scoring on held-back names is the honest test. Shuffle the names with a fixed seed, train (or count) on the first 90%, and work out the loss on the last 10%. That's the **validation loss**.

| Score | What to record | Reference | Passes if |
|---|---|---|---|
| Validation loss | the loss on the held-back 10% | about 2.45 | 2.5 or less |
| Training time | seconds for 100 steps of the neural version on all the pairs | about 1.5 s on a recent graphics card (an estimate) | 60 s or less |

A bigram can't do much better than about 2.45, because it only ever sees one letter. Lesson 0.4 trains a bigger model that looks at three letters and gets to about 2.1.

## Words you'll see {#glossary}
> A short list of the new words in this lesson's cards, challenges and scores.

- **Bigram**: a pair of neighbouring letters, and the model that guesses each letter from the one before it.
- **Count table (N)**: the 27 by 27 table of how often each letter followed each other letter.
- **P**: the table of chances: each row of N divided by its total.
- **P(b | a)**: the chance of "b" given that the current letter is "a". Row "a", column "b".
- **Smoothing**: adding 1 to every count, so no pair has a chance of 0.
- **Sampling**: picking at random, using the chances.
- **Multinomial**: PyTorch's weighted random pick, `torch.multinomial`.
- **Generator, seed**: a source of random numbers, and the starting number that makes its picks repeatable.
- **`argmax`**: the position of the biggest number. Always taking it gives the same name every time.
- **`.item()`**: turns a one-number tensor into a plain Python number.
- **Neural network**: a model with adjustable numbers (weights) that it learns from data.
- **Weights (W)**: the numbers the network learns. Here, a 27 by 27 table.
- **One-hot**: a list of zeros with a single 1 at an item's position.
- **Logits**: the raw scores, before softmax.
- **Softmax**: exp each score, then divide by the row total. Turns scores into chances.
- **Gradient descent**: training by stepping the weights against their grads, again and again.
- **Learning rate**: how big each step is. 50 in this lesson.
- **`torch.no_grad()`, `.data`**: two ways to change weights without PyTorch tracking it.
- **`nan`**: "not a number". What a loss turns into when training blows up.
- **Validation loss**: the loss on names the model never trained on.
- **Training time**: this lesson's speed score: seconds for 100 training steps on all the pairs.
