---
gist: A small draft model guesses a few tokens, the big model checks all of them in one pass, and you keep what it agrees with. One user gets text faster, and an exact accept/reject rule keeps the output identical to the big model's.
minutes: 17
---

## What this lesson is for {#why}
> For one user, the GPU spends most of each decode step waiting on memory. Speculative decoding fills that idle time by checking several guessed tokens per pass.

In 5.1 you worked out that one user's decode step on your 8B model takes about 17 ms, almost all of it spent reading weights. The math units sit idle about 98% of the time. Batching more users uses that idle time, but what if there's only one user, say in a chat app?

**Speculative decoding** uses the idle time for that one user. A small, fast **draft** model guesses the next few tokens. The big **target** model then scores all the guesses in a *single* forward pass, which costs about the same as making one token, and you keep the guesses it agrees with. If the draft guesses well, each expensive pass yields several tokens instead of one.

The surprising part: the output is **exactly** what the target alone would produce, in greedy mode and in sampling mode. In this lesson you build the greedy version and check it, write the accept/reject rule that keeps sampling exact, work out when it pays off, and then try a 1B draft with your 8B model on the GPU.

## What you need from before {#before}
> Logit row j predicts token j + 1, one forward pass scores every position at once, and decode for one user is memory-bound.

Three facts from earlier lessons:

- **One pass, every position.** A transformer's forward pass over `n` tokens returns logits of shape `[n, vocab]`. Row `j` is the model's prediction for the token at position `j + 1`. Training used this (1.x), and so does prefill.
- **Memory-bound decode** (5.1). A pass costs about the same whether it scores 1 position or 5, because reading the weights dominates.
- **Greedy vs sampling** (2.3). Greedy takes the top token. Sampling draws from the probabilities, usually after temperature or top-p.

## Guess, then check {#idea}
> The draft proposes k tokens one at a time; the target scores all k + 1 positions in one pass; you keep the guesses up to the first disagreement, plus one token from the target.

If you know how a CPU runs code, this is **speculative execution**: guess which way a branch goes, run ahead, and throw the work away if the guess was wrong. Here the guess is text.

One round with k = 3:

1. The draft generates 3 tokens, one at a time. Each guess sees the guesses before it.
2. The target runs **once** over the text so far plus the 3 guesses. That gives its own choice at each guessed position, plus one position beyond.
3. Walk the guesses left to right. Keep each one the target agrees with. At the first disagreement, take the target's token instead and stop.
4. If all 3 survive, the extra position gives a **bonus** token for free.

Every round yields between 1 and k + 1 tokens for one target pass. It can never yield zero: even a bad guess gets replaced by the target's own token.

## Greedy verification, step by step {#greedy}
> In greedy mode, a guess survives when it equals the target's argmax at that position. The first mismatch is replaced by the target's choice, and the round ends.

Say the text so far has `n` tokens and the draft proposed `[5, 2, 7]`. The target's pass over all `n + 3` tokens gives rows `n − 1` to `n + 2`. Each row's argmax is what the target itself would put at the next position:

| Position | Draft guessed | Target's choice | Result |
|---|---|---|---|
| n | 5 | 5 | keep 5 |
| n + 1 | 2 | 2 | keep 2 |
| n + 2 | 7 | 1 | take 1, stop |
| n + 3 | (none) | 4 | not reached |

The round appends `[5, 2, 1]`: three tokens from one target pass, and exactly what greedy decoding would have produced.

Why stop at the first mismatch? The target's row for position `n + 3` was computed with the draft's `7` in the text. Once you replace `7` with `1`, that row answers a question greedy never asks, so it's useless. The tests catch three tempting mistakes: keeping the draft's token at the mismatch, carrying on past it, and reading the wrong row. The row that checks the guess at position `n + i` is `n − 1 + i`, one row earlier.

## Counting target passes {#passes}
> Correct output is only half the job: each round must cost exactly one target pass, or you've rebuilt plain greedy.

The browser challenge uses two tiny models, a target and a blurred copy as the draft, and counts calls. For 24 new tokens, plain greedy needs 24 target passes, while the speculative version with k = 4 needs 14 and produces the same tokens.

Three ways to lose the speedup while still getting the right text:

- **Calling the target once per guessed position**, instead of once over all of them. That's plain greedy with extra steps.
- **Forgetting the bonus token.** When all k guesses survive, the last row already holds the target's next token. Skip it and you pay a whole extra pass to get it.
- **A non-autoregressive draft.** If every guess only sees the original text, the draft proposes the same token k times, and most get rejected. Each guess has to see the guesses before it.

## Sampling: matching argmax isn't enough {#sampling}
> When you sample, the goal is that every token follows the target's probabilities exactly, not that it matches one particular choice.

With temperature above 0, there's no single right token. The target has a probability `p(x)` for every token `x`, and correct output means each token is drawn with exactly those probabilities. The draft has its own probabilities `q(x)`, and its guesses are drawn from `q`.

If you simply kept every draft guess, the text would follow `q`, the draft's distribution, which is wrong. If you kept a guess only when it happened to be the target's top token, you'd lose the randomness. The paper's rule does something cleverer.

## The accept/reject rule {#rule}
> Keep a drafted token x with probability min(1, p(x) / q(x)). If you reject it, sample a replacement from max(0, p − q), rescaled, and stop.

Take one position with 4 possible tokens:

```
token     0    1    2    3
p      0.10 0.50 0.30 0.10
q      0.40 0.40 0.10 0.10
```

- The draft proposes token 0 a lot (0.40), but the target wants it rarely (0.10). Keep it with probability 0.10 / 0.40 = **25%**.
- Token 2 is under-proposed (0.10 vs 0.30). The ratio is 3, so keep it **always**.
- To decide, draw a uniform random number `u` in [0, 1) and keep the token when `u` is below the ratio.

When a guess is rejected, the replacement must add back exactly what the draft under-proposed:

```
p - q     -.30  .10  .20  .00
max(0,·)   0   .10  .20   0
rescaled   0   1/3  2/3   0
```

Check it adds up for token 2. It's proposed 10% of the time and always kept, which gives 0.10. A rejection happens 30% of the time overall, and then token 2 is picked 2/3 of the time, which gives 0.20. The total is 0.30, exactly `p(2)`. The same works for every token. That's the theorem in Leviathan et al.: the output follows `p` exactly.

The tests check the classic slips: comparing `p > q` without a random draw (deterministic, so it drifts), the ratio upside down (`q / p`), ignoring `q`, forgetting to clamp the negative entries (sampling fails), replacing from `p` instead of the residual, and taking the argmax instead of sampling. Every random draw goes through a seeded **generator**, so a run repeats exactly and the tests can count frequencies.

## Expected tokens per round {#expected}
> If each guess survives with probability α, a round yields 1 + α + α² + … + α^k tokens on average.

Call the chance a guess survives the **acceptance rate**, α. A round stops at the first rejection, so guess 2 only counts if guess 1 survived too:

| Guess | Counts if | Probability |
|---|---|---|
| 1 | guess 1 survives | α |
| 2 | guesses 1–2 survive | α² |
| 3 | guesses 1–3 survive | α³ |
| target's token | always | 1 |

Add the column. With α = 0.75 and k = 4:

```
1 + .75 + .5625
  + .4219 + .3164
  = 3.05 tokens per pass
```

There's also a closed form: (1 − α^(k+1)) / (1 − α). Two mistakes to avoid: forgetting the target's own token (the 1), and multiplying α × k as if every guess were checked separately.

## When it pays off {#speedup}
> A round costs k draft passes plus one target pass. Speedup is tokens per round divided by that cost, and it peaks at some best k.

Say a draft pass costs `c` of a target pass. A 1B draft for an 8B target is roughly c = 0.15 to 0.2, since it reads far fewer weights. A round then costs k × c + 1 target passes, and plain decoding makes 1 token per target pass, so:

```
speedup = tokens per round
          / (k × c + 1)
```

With α = 0.75 and c = 0.15:

| k | Tokens per round | Cost | Speedup |
|---|---|---|---|
| 1 | 1.75 | 1.15 | 1.52× |
| 2 | 2.31 | 1.30 | 1.78× |
| 4 | 3.05 | 1.60 | 1.91× |
| 8 | 3.70 | 2.20 | 1.68× |

```chart
kind: line
title: Speedup against draft length k (α = 0.75, c = 0.15)
x: { label: draft length k, values: [1, 2, 4, 8] }
y: { label: speedup (×), min: 0, max: 2 }
series:
  - { name: speedup, values: [1.52, 1.78, 1.91, 1.68] }
highlight: { x: 4, label: "best k, 1.91×" }
threshold: { y: 1, label: "plain decoding, 1×" }
caption: Speedup climbs while extra guesses still pay for themselves, peaks at k = 4, then falls as each guess costs more than it earns.
```

Past the best k, each extra guess costs more than it is likely to earn. A poor draft can even be slower than no draft: with α = 0.1 and k = 1 a round yields 1.1 tokens for a cost of 1.15. The output stays exact either way. Only the speed changes.

## Choosing a draft {#draft}
> The draft must use the target's tokenizer and predict text like the target does. Acceptance is everything.

- **Same tokenizer.** Guesses are token ids, so both models must use the same vocabulary. Your 4A.5 model keeps Llama 3.1's tokenizer, so Llama 3.2 1B can draft for it. Your 4B model's extended Mongolian vocabulary has no small sibling. For it, **prompt-lookup** drafting proposes tokens copied from earlier in the prompt and needs no draft model at all.
- **Same kind of text.** The 1B draft knows English far better than Mongolian, so α drops on Mongolian prompts and so does the best k. Fine-tuning the draft on your corpus raises it.
- **Small batches only.** Speculative decoding spends spare compute. At large batch sizes the GPU is already compute-bound (5.1's roofline), and the extra positions stop being free.

## The GPU run {#gpu-run}
> You'll sweep k with a 1B draft and your 8B target, find the best k, and check greedy outputs are unchanged.

The course repo's `spec_decode.py` runs in about 30 minutes. It measures plain decoding first, then speculative decoding at k = 1, 2, 3, 4 and 6, printing α, tokens per round, tokens per second and the speedup for each. Then it checks 50 greedy outputs against the target alone.

What healthy output looks like, from the lesson's replay. The replay is **illustrative sample data written for the course, not a real recording**:

- On Mongolian prompts α is about 0.6, and the best k is small (2), at about 1.4×.
- At k = 6 the speedup falls back toward 1×: drafting costs more than it saves.
- The greedy check matches on 50 of 50 prompts. A rare mismatch at a near-tie can happen, because the target scores k + 1 positions at once and fp16 rounds a little differently.
- On English prompts the same draft does much better.

You record the greedy match percentage and the speedup at your best k.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Speculative decoding**: draft several tokens cheaply, verify them all in one pass of the big model.
- **Target model**: the big model whose output you want, here your 8B.
- **Draft model**: the small, fast model that guesses, here Llama 3.2 1B.
- **Proposal, guess, draft token**: a token the draft suggested.
- **k (γ in the paper)**: how many tokens the draft guesses per round.
- **Round**: k draft passes plus one target pass.
- **Verify**: score all the guesses in one target pass and decide which to keep.
- **Bonus token**: the target's extra token when every guess survives.
- **Greedy**: always taking the top token.
- **p, q**: the target's and the draft's probabilities for the next token.
- **Accept/reject rule**: keep a guess with probability min(1, p/q), otherwise sample from max(0, p − q).
- **Residual**: max(0, p − q), what the target wants more than the draft offered.
- **Generator, seed**: a source of random numbers that repeats exactly from the same seed.
- **Acceptance rate (α)**: the chance a guess survives.
- **Expected tokens per round**: 1 + α + … + α^k.
- **c (draft cost)**: one draft pass as a fraction of a target pass.
- **Speedup**: tokens per target-pass-equivalent, compared with plain decoding's 1.
- **Prompt-lookup drafting**: proposing tokens copied from the prompt, with no draft model.
- **Lossless**: the output distribution is exactly the target's.
- **Illustrative**: sample data written for the course, not a recording of a real run.
