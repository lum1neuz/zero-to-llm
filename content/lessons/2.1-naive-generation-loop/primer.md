---
gist: A language model only ever predicts the next token, so writing text means running it in a loop that picks one token, appends it and runs again, and this lesson builds that loop with greedy choice and then temperature.
minutes: 15
---

## What this lesson is for {#why}
> Your GPT from phase 1 can score the next token, but it can't write anything on its own. This lesson builds the loop that turns it into a text generator.

In 1.3 and 1.4 you trained a model that, given some tokens, outputs a guess for what comes next. That's all a language model does. A story is hundreds of those guesses in a row, where each guess becomes part of the input for the next one.

This lesson writes that loop twice:

1. **Greedy**: always take the single most likely token.
2. **Temperature sampling**: draw the token at random, in proportion to how likely the model thinks it is, with a knob for how adventurous to be.

It's also the slow, honest baseline for the rest of phase 2. Lesson 2.2 makes it fast with a KV cache, 2.3 makes the sampling smarter, 2.4 runs many prompts at once and 2.5 measures where the time goes. Every later speed-up has to produce exactly what this loop produces.

## What you need from before {#before}
> Token ids in, one row of logits per position out. Softmax turns a row of logits into probabilities.

A quick refresher from phases 0 and 1:

- **Tokens and ids.** Text is split into tokens, and each token has an integer id. The tiny model in this lesson uses characters: `"the "` becomes four ids.
- **Logits.** For each position, the model outputs one score per vocabulary entry. Bigger means "more likely next". They aren't probabilities yet: they can be negative and don't add up to 1.
- **Softmax** turns a row of logits into probabilities that are all positive and sum to 1.
- **The causal mask.** In training, position t only sees positions 0…t. That's what lets one forward pass make a prediction at every position at once.

Your 1.x model takes `[B, T]` and returns `[B, T, V]`. In this lesson the batch dimension is dropped to keep shapes short, so the model takes `[T]` and returns `[T, V]`. Lesson 2.4 brings `B` back.

## One token at a time {#autoregressive}
> Run the model, take its guess for the next token, glue that token onto the input, and run it again.

This is called **autoregressive** generation: the model's own output feeds back in as input.

```
step 1:  "the "      → guess "c"
step 2:  "the c"     → guess "a"
step 3:  "the ca"    → guess "t"
step 4:  "the cat"   → guess " "
```

Each step adds exactly one token. After 60 steps from a 4-token prompt, the sequence is 64 tokens long. In Java terms, it's a `for` loop that keeps growing an `ArrayList<Integer>` of ids, except here the list is a tensor.

## Which row to read {#last-row}
> The logits have one row per position, and row t is the guess for the token after position t. Only the last row predicts a token you don't have yet.

Say the sequence has 3 tokens and the vocabulary has 3 entries. The model returns a `[3, 3]` table:

```
         id 0  id 1  id 2
row 0  [ 0.1,  2.0,  0.3 ]
row 1  [ 1.5,  0.2,  0.4 ]
row 2  [ 0.0,  0.1,  3.0 ]
```

| Row | Predicts the token after… | Useful now? |
|---|---|---|
| 0 | position 0 | No, position 1 already exists |
| 1 | position 1 | No, position 2 already exists |
| 2 (the last) | position 2 | **Yes**, this is the new token |

Rows 0 and 1 were essential in training, where every position has a target to check against. At generation time they're wasted work. The newest prediction always lives in the last row, which Python reads as index `-1`.

Two classic mistakes:

- **Reading row 0 every step.** You get the same prediction over and over, because row 0 only ever sees the first token.
- **Taking argmax of the whole table.** `argmax` on a 2-D tensor flattens it first, so you get a position in the 9-cell table (here 8, where 3.0 sits), not a token id. With a real vocabulary that number can be far larger than the vocabulary.

## Greedy: take the top token {#greedy}
> Greedy decoding picks the id of the largest logit. It's deterministic, and it easily gets stuck repeating itself.

In row 2 above, the largest logit is 3.0 at position 2, so greedy picks token id 2. Note the difference between the *value* (3.0) and its *position* (2). The position is the token id, and `argmax` is the function that returns positions.

Greedy has no randomness. Give it the same context and it makes the same choice, every time. If a phrase makes itself likely again, greedy can never leave it. The tiny model in this lesson, started from `"the "`, writes:

```
the the the the the the the …
```

That isn't a bug in your loop. It's the model's honest top choice, repeated. Sampling, next, is the way out.

## Growing the sequence {#append}
> Tensors have no `append`. You build a new, longer tensor by joining the old one and the new id with `torch.cat`, and both pieces need at least one dimension.

`argmax` on one row returns a **0-d tensor**: a single number with shape `[]`. `torch.cat` joins tensors *along* a dimension, so it refuses a piece that has none.

| Thing | Shape | Can `torch.cat` join it to `[T]`? |
|---|---|---|
| `idx` | `[T]` | it's the base |
| the argmax result | `[]` | No: no dimension to join along |
| the same id as a 1-element tensor | `[1]` | Yes, giving `[T + 1]` |

So the new id has to become a 1-element tensor before it's joined. The error you'd see otherwise is `zero-dimensional tensor (at position 1) cannot be concatenated`.

Two more mistakes the tests catch:

- **Replacing instead of appending.** If `idx` becomes just the new id, the model sees one token next step and forgets the story.
- **Running the model on only the last token.** A bigram table could manage, but a transformer's next token depends on the whole context. Feed it all of `idx`. Keeping the old tokens' work cheaply is exactly what 2.2's cache does.

## From logits to probabilities {#softmax}
> Softmax exponentiates each logit and divides by the total, so bigger logits get more of the probability.

Take three logits, `[2, 1, 0]`:

| Logit | e^logit | ÷ total (11.11) |
|---|---|---|
| 2 | 7.39 | 0.67 |
| 1 | 2.72 | 0.24 |
| 0 | 1.00 | 0.09 |

The probabilities sum to 1. Greedy would always take the first token. **Sampling** takes it 67% of the time, the second 24% and the third 9%. That's like a weighted lottery where each token holds as many tickets as its probability says.

## Temperature {#temperature}
> Divide the logits by a temperature T before softmax. T below 1 sharpens the distribution toward greedy; T above 1 flattens it toward random.

Dividing changes the *gaps* between logits, and softmax only cares about gaps:

| T | Logits ÷ T | Probabilities |
|---|---|---|
| 0.5 | [4, 2, 0] | [0.87, 0.12, 0.02] |
| 1.0 | [2, 1, 0] | [0.67, 0.24, 0.09] |
| 2.0 | [1, 0.5, 0] | [0.51, 0.31, 0.19] |

At T = 0.5 the top token gets most of the probability. As T heads toward 0 it becomes greedy. You can't actually use T = 0, because that divides by zero, which is why greedy is its own function in the challenge.

Three mistakes the tests recognise from the draws themselves:

- **Multiplying by T.** At T = 0.5 that halves the gaps and *flattens* the distribution, the opposite of what you want.
- **Dividing after softmax.** It only rescales the probabilities, and the sampler renormalises them, so the temperature has no effect at all.
- **Using `log_softmax`.** Log-probabilities are all zero or negative, and a sampler needs probabilities, so it refuses them.

## Drawing a token, reproducibly {#sampling}
> `torch.multinomial` draws ids in proportion to a row of probabilities. Passing it a seeded generator makes the draws repeatable.

`torch.multinomial(probs, 1)` returns one id, as a 1-element tensor of shape `[1]`. Ask for many draws from `[0.67, 0.24, 0.09]` and about two thirds will be id 0.

Random draws come from a **generator**, a stream of pseudo-random numbers. Seed it with `torch.Generator().manual_seed(42)` and you get the same stream every run, so the same text. That's how you compare strategies fairly and debug one bad sample. If your function ignores the generator it was given, two runs with the same seed come out different, and the tests notice.

The browser's mini-torch is seeded and repeatable, but its numbers aren't the same as real PyTorch's. The same seed on your GPU gives a different, equally repeatable story.

## The cost of no cache {#cost}
> Each step re-runs the model on the whole sequence, so the total work grows with the square of the length.

From a 4-token prompt, step 1 processes 4 positions, step 2 processes 5, and step 60 processes 63:

```
4 + 5 + … + 63
= 60 × (4 + 63) / 2
= 2010 positions
```

That's 2,010 positions of work to write 60 characters, which your script prints as `positions processed`. Double the length and the work roughly quadruples. Most of it recomputes things that haven't changed, and lesson 2.2 stops doing that.

## On your GPU {#gpu}
> You'll run the same loop on your 30M TinyStories model. The recording shows time per token climbing as the story grows. It's illustrative sample data, not a real run.

The replay step shows `generate.py` writing 512 tokens greedily with no cache. Healthy output looks like this:

- Time per token climbs from about 3 ms to about 7 ms between the first and last 64 tokens.
- The total comes to around 216 tokens per second.
- Greedy output matches the course repo's reference `generate()` on all 512 tokens.

The numbers are sample data written for the course, labelled as such in the app. Yours will differ with your GPU. You record two metrics: the percentage of greedy tokens that match the reference (aim for 99% or more) and your tokens per second.

Three things to remember on the GPU:

- Wrap the loop in `torch.no_grad()`. Otherwise autograd keeps every step's activations for a backward pass that never comes.
- Your model has a fixed context length, `block_size`. Once the sequence is longer, feed only the last `block_size` ids.
- With a batch dimension, the last row is `logits[:, -1, :]`.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Autoregressive generation**: producing text one token at a time, feeding each new token back in as input.
- **Logits**: the model's raw scores, one per vocabulary entry per position. Bigger means more likely.
- **Vocabulary (V)**: every token the model knows. The tiny model's vocabulary is 15 characters.
- **Greedy decoding**: always picking the id of the largest logit.
- **argmax**: returns the *position* of the largest value, which for a row of logits is a token id.
- **0-d tensor**: a tensor holding one number, with shape `[]`.
- **torch.cat**: joins tensors along an existing dimension into a longer one.
- **Softmax**: turns logits into probabilities that are positive and sum to 1.
- **Sampling**: drawing the next token at random, in proportion to its probability.
- **Temperature (T)**: the number the logits are divided by before softmax. Low is sharp, high is flat.
- **torch.multinomial**: draws ids from a row of probabilities.
- **Generator / seed**: the source of random numbers, and the number that makes it repeat.
- **Positions processed**: how many positions the model has run in total. Without a cache it grows quadratically.
- **Tokens per second (tok/s)**: how fast generation runs, this lesson's systems metric.
- **Sample match**: the share of greedy tokens your loop produces that agree with a reference implementation, this lesson's quality metric.
- **block_size**: the longest sequence your model can take at once.
