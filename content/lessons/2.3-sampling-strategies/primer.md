---
gist: Greedy text repeats itself and plain sampling wanders into nonsense, so top-k and top-p cut the unlikely tokens before you draw, and distinct-2 measures how repetitive the result is.
minutes: 14
---

## What this lesson is for {#why}
> Sampling fixed greedy's loops but brought in a new problem: now and then it picks a token the model barely believes in. This lesson cuts those tokens out before the draw.

In 2.1 you saw both failures. Greedy wrote "the the the…" forever. Temperature sampling escaped the loop, but at T = 1.5 it drifted into noise, because every so often it drew a token from the long tail of unlikely ones.

Here you write the two standard fixes:

1. **Top-k**: keep only the k most likely tokens.
2. **Top-p**, also called nucleus sampling: keep the most likely tokens until their probabilities add up to p.

Then you measure repetition with **distinct-2** and compare four strategies on the same model and seed. Every chat model you'll use applies something like this on every token, and phase 5's serving systems expose these same settings.

## What you need from before {#before}
> From 2.1: softmax turns logits into probabilities, temperature divides the logits first, and torch.multinomial draws an id with a seeded generator.

The pipeline for one sampled token, as of the end of 2.1:

```
logits [V]
 → ÷ temperature
 → softmax → probs [V]
 → multinomial → one id
```

This lesson adds one stage between temperature and softmax: a **filter** that sets the logits of unwanted tokens to `-inf`. The start code's `sample` function already has a slot for it, called `keep`.

## Two ways to fail {#two-failures}
> Picking the top token every time is repetitive. Drawing from everything is incoherent. Good text sits in between.

From the tiny model in this lesson, same prompt, same seed:

| Strategy | Output starts |
|---|---|
| greedy | `the the the the the the` |
| T = 1.0, no filter | `the ra atndogsr t t the a` |
| top-p 0.9 | `the ra athe the t t the a the` |

The tiny model is a bigram table with one attention head, so none of these read well. The point is the shape of the failure. Greedy repeats, unfiltered sampling makes non-words like "atndogsr", and the filtered version stays closer to real words.

## The tail {#tail}
> Each unlikely token is rare on its own, but there are thousands of them, so together they can hold a big share of the probability.

Imagine a vocabulary of 4,096 tokens and one step's probabilities:

| Tokens | Probability |
|---|---|
| top 1 | 0.45 |
| top 2 | 0.20 |
| top 3 | 0.10 |
| the other 4,093 | 0.25 together |

No single tail token is likely. But a quarter of the probability sits in the tail, so roughly one draw in four lands there. The model's estimates for those rare tokens are also its least reliable. One bad pick mid-sentence, and every later token builds on it.

## Cutting a token: -inf, not 0 {#neg-inf}
> To remove a token, set its logit to -inf. Softmax turns -inf into exactly 0. A logit of 0 is an ordinary score.

Softmax exponentiates each logit, and e^(-inf) is 0. So a token with logit `-inf` gets probability 0 and can never be drawn, and the rest share the probability among themselves:

```
logits  [2, 1, -inf, -inf]
e^x     [7.39, 2.72, 0, 0]
probs   [0.73, 0.27, 0, 0]
```

Setting a logit to 0 is a classic mistake. e⁰ = 1, so the "removed" token still gets a share. On the test row, five tokens set to 0 still held 22% of the probability.

## Top-k {#top-k}
> Find the k-th largest logit and use it as a cutoff: everything strictly below it becomes -inf. The row keeps its shape.

Take the logits `[1, 4, 2, 3]` and k = 2:

```
sorted, largest first: 4, 3, 2, 1
2nd largest (cutoff):  3
keep ≥ 3:  [-inf, 4, -inf, 3]
```

The two largest stay where they were, with their values unchanged. The output has one logit per token, so it still lines up with the vocabulary. `logits.topk(k)` helps here: it returns the k largest *values* (sorted, largest first) and their *indices* (positions), as a pair.

Mistakes the tests recognise from what you keep:

| What you did | What happens |
|---|---|
| Cutoff = the largest value | Only 1 token survives |
| Cut values *at* the cutoff too | k − 1 survive |
| Cut the large side | You keep the smallest |
| Used an index as the cutoff | Positions compared with logits: nonsense |
| Returned the k values only | The row shrinks to k |

## Top-p {#top-p}
> Sort the tokens by probability, add them up from the top, and keep tokens until the total reaches p. The token that crosses p stays in.

Here are six sorted probabilities and p = 0.9:

| Token | Prob | Total above it | Running total | Kept? |
|---|---|---|---|---|
| 1st | 0.50 | 0.00 | 0.50 | yes |
| 2nd | 0.20 | 0.50 | 0.70 | yes |
| 3rd | 0.15 | 0.70 | 0.85 | yes |
| 4th | 0.08 | 0.85 | 0.93 | yes |
| 5th | 0.04 | 0.93 | 0.97 | no |
| 6th | 0.03 | 0.97 | 1.00 | no |

The useful column is **"total above it"**. A token is cut once the tokens ranked above it already cover p. The 4th token has only 0.85 above it, which isn't enough yet, so it stays. Its own probability takes the total to 0.93.

Three mistakes to avoid:

- **Cutting by the running total itself** (including the token). That cuts the 4th token too. Worse, if the top token alone is above p, everything gets cut, and softmax over a row of `-inf` gives NaN.
- **Adding up in vocabulary order.** The running total only means something down the *sorted* list.
- **Cutting by each token's own probability.** That's a different rule. Top-p adds probabilities up.

After deciding which sorted positions to cut, the cut has to be mapped back to vocabulary order, because the logits you return are in vocabulary order. The stub does that step for you with `argsort`, which undoes a sort.

## Top-k or top-p? {#compare}
> Top-k keeps a fixed number of tokens. Top-p keeps however many it takes to cover p, so it shrinks when the model is sure and grows when it isn't.

| Model's state | Top-k 50 keeps | Top-p 0.9 keeps |
|---|---|---|
| Sure: top token 0.95 | 50 | 1 |
| Torn: 30 tokens near 0.03 each | 50 | about 30 |

When the model is sure, top-k still lets 49 long shots into the draw. That adaptivity is why top-p usually reads better, and why libraries often combine both.

**Order matters.** Temperature comes first, then the filter. Temperature changes the probabilities that top-p adds up, so with T > 1 the nucleus grows. If you swapped the order, the nucleus would ignore the temperature.

## Measuring repetition: distinct-2 {#distinct}
> Take every pair of neighbouring tokens. distinct-2 is the number of different pairs divided by the number of pairs.

Take the ids `[1, 2, 1, 2, 1, 2]`. Six ids give five neighbouring pairs:

```
(1,2) (2,1) (1,2) (2,1) (1,2)
different: (1,2) and (2,1) → 2
distinct-2 = 2 / 5 = 0.4
```

Text that loops scores low. Varied text scores close to 1. Note the denominator: n ids make n − 1 pairs, not n.

In Python, a `set` keeps one copy of each distinct item, like Java's `HashSet`. Its items have to be hashable, so the pairs must be **tuples**, not lists, because lists can change. `zip(ids, ids[1:])`, which you met in lesson 0.0, gives tuples.

On the tiny model, greedy scores 0.08, unfiltered T = 1.0 scores 0.62, and top-k and top-p land in between. **Higher isn't always better.** Random noise has very high distinct-2. Read the samples as well as the number.

## On your GPU {#gpu}
> You'll sample 100 stories per strategy from your 30M model and compare distinct-2 and sampling time. The recording is illustrative sample data, not a real run.

The replay shows `sample.py --compare`. Healthy output looks like:

- Greedy at about 0.01: all 100 samples are the same story.
- Top-p 0.9 around 0.41, with readable, varied stories.
- T = 1.5 around 0.83, flagged as word salad.
- Sampling time for top-p around 86 µs per token, about 4% of a roughly 2 ms decode step.

The numbers are sample data, labelled as such in the app. You record two metrics for top-p 0.9: distinct-2 over the 100 samples (at least 0.25) and the sampling time per token in microseconds.

On the GPU, filter a whole batch at once: logits are `[B, V]`, and `topk` and `sort` work along the last dimension. Create the generator on the GPU too, with `torch.Generator(device="cuda")`.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Tail**: the many low-probability tokens at the bottom of the distribution.
- **Top-k**: keep the k most likely tokens and cut the rest.
- **Top-p / nucleus sampling**: keep the most likely tokens until their probabilities sum to p.
- **Nucleus**: the set of tokens top-p keeps.
- **-inf**: negative infinity. As a logit, it gives probability exactly 0.
- **topk**: returns the k largest values and their indices.
- **Running total (cumsum)**: each entry is the sum of everything up to and including it.
- **argsort**: the positions that would sort a tensor. Taking `argsort` of a sort order undoes the sort.
- **masked_fill**: replaces entries where a mask is True with a given value.
- **Greedy**: always the top token. Covered in 2.1.
- **Temperature**: divides the logits before softmax. Covered in 2.1.
- **Seeded generator**: a repeatable source of random numbers, so the same seed gives the same text.
- **distinct-2**: different neighbouring pairs ÷ all neighbouring pairs, this lesson's quality metric.
- **distinct-1**: the same idea for single tokens.
- **Tuple**: a fixed, unchangeable sequence like `(1, 2)`. Tuples can go in a set; lists can't.
- **Sampling overhead**: time spent choosing the token, not running the model, this lesson's systems metric (µs per token).
