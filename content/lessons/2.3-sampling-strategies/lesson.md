---
id: "2.3"
title: Sampling strategies
summary: Cut the unlikely tail before you sample. Write top-k and top-p, then measure how repetitive each strategy's text is with distinct-2.
phase: "2"
hours: 3
type: build
status: ready
protected: false
cut: null
prerequisites:
  - "2.1"
papers:
  - title: The Curious Case of Neural Text Degeneration
    authors: Holtzman et al.
    year: 2019
    url: https://arxiv.org/abs/1904.09751
quality_metric:
  id: distinct_2
  label: distinct-2 on 100 samples
  unit: ratio
  note: Unique neighbouring token pairs ÷ all pairs, pooled over 100 top-p 0.9 samples of 200 tokens from one prompt. Higher means less repetition. Very high usually means noise, so read the samples too.
  inputs:
    - { id: distinct_2, label: "distinct-2, 100 top-p 0.9 samples", hint: Printed by sample.py --compare }
  compute: value
  reference: 0.41
  pass: { min: 0.25 }
systems_metric:
  id: sampling_us
  label: Sampling overhead per token
  unit: µs
  note: Time spent choosing the token (temperature, top-p, multinomial), not the forward pass.
  inputs:
    - { id: sampling_us, label: top-p sampling time per token (µs), hint: Printed by sample.py --compare }
  compute: value
  reference: 86
  pass: { max: 500 }
quiz:
  - id: confident_model
    q: The model is nearly sure of the next token, with probability 0.95. How many tokens do top-p 0.9 and top-k 50 keep?
    answer: 1
    options:
      - text: Both keep 50
        why: Top-p doesn't have a fixed size. Its cut depends on the probabilities, not on a count.
      - text: Top-p keeps 1, top-k keeps 50
        why: The top token alone covers 0.9, so top-p stops there. Top-k always keeps 50, so 49 tokens with 0.05 between them stay in play. That adaptivity is why top-p usually reads better.
      - text: Top-p keeps 50, top-k keeps 1
        why: It's the other way round. Top-k's size is fixed at k, and top-p shrinks when the model is confident.
  - id: why_seed
    q: Why pass a seeded torch.Generator into the sampler?
    answer: 0
    options:
      - text: So the same seed gives the same text, and anyone can reproduce your samples
        why: The draws come from the generator's stream. Same seed, same stream, same story. That's how you compare strategies fairly and debug one bad sample.
      - text: It makes sampling faster
        why: Speed is the same. The generator only decides which random numbers come out.
      - text: It makes the text more varied
        why: The seed doesn't change how varied the text is. Temperature, top-k and top-p do.
  - id: temperature_first
    q: Sampling applies temperature first, then top-p. What changes if you swap the order?
    answer: 2
    options:
      - text: Nothing. They commute
        why: They don't. Top-p adds up probabilities, and temperature changes those probabilities.
      - text: Top-p stops working
        why: It still keeps a nucleus. It just keeps a different one.
      - text: The nucleus is chosen from the untempered probabilities, so it's a different set of tokens
        why: With temperature first, T > 1 flattens the distribution and top-p keeps more tokens. Swapped, the nucleus ignores T. Libraries usually apply temperature first.
---

## Goal

Sampling at temperature 1 sometimes picks a token from the long, unlikely tail, and one bad token can derail a story. Write the two standard cuts: **top-k** keeps the k most likely tokens, and **top-p** (nucleus) keeps the smallest set whose probability adds up to p. Then measure repetition with **distinct-2** and compare greedy, temperature, top-k and top-p on the tiny model and on your 30M model.

## Learn

```card
type: concept
id: the_tail
eyebrow: Why cut
title: Thousands of unlikely tokens add up.
visual:
  kind: bars
  max: 0.5
  bars:
    - { label: top 1, value: 0.45, text: "0.45" }
    - { label: top 2, value: 0.20, text: "0.20" }
    - { label: top 3, value: 0.10, text: "0.10" }
    - { label: "the other 21,501", value: 0.25, text: "0.25 together" }
body: |
  Each token in the tail is unlikely on its own, but together they hold a quarter of the probability, so at T = 1 one draw in four lands somewhere in the tail. **Top-k** keeps the k most likely tokens. **Top-p** keeps the most likely tokens until their probabilities add up to p. Both set the rest to `-inf`, so softmax gives them exactly 0.
```

```card
type: predict
id: predict_top_k
eyebrow: Predict the output
title: What does top_k return?
code: |
  logits = torch.tensor([1.0, 4.0, 2.0, 3.0])
  print(top_k(logits, 2))
options:
  - text: "tensor([-inf, 4., -inf, 3.])"
    correct: true
    why: The 2 largest logits, 4 and 3, keep their places and values. The rest become -inf, so softmax gives them 0.
  - text: "tensor([4., 3.])"
    why: That's `logits.topk(2).values`. The sampler needs one logit per token, so top_k keeps the shape.
  - text: "tensor([-inf, 4., 2., 3.])"
    why: That keeps 3 tokens. With k = 2 the cutoff is the 2nd largest, 3.0, so 2.0 goes too.
  - text: "tensor([1., -inf, 2., -inf])"
    why: That keeps the 2 smallest. Top-k keeps the largest.
hint: Find the 2 biggest numbers. They stay where they are, and everything else becomes -inf.
```

```card
type: fill_blank
id: top_p_rule
eyebrow: Fill the blanks
title: Complete the top-p cut.
code: |
  sorted_probs, order = probs.sort(descending=True)
  cum = sorted_probs.[[0]](dim=-1)
  cut_sorted = cum - sorted_probs [[1]] p
tokens: [cumsum, sum, ">=", "<", softmax, "=="]
answer: [cumsum, ">="]
tips:
  sum: "`sum` gives one total. You need a running total, one value per token."
  "<": That cuts the most likely tokens and keeps the tail.
  softmax: The probabilities are already there. Top-p adds them up.
  "==": Float sums almost never hit p exactly, so nothing would be cut.
why_ok: "`cum - sorted_probs` is the probability of the tokens ranked *above* each one. Once that reaches p, the token is cut. The token that crosses p stays, so the kept set always covers p."
why_wrong: "Check both blanks: you need a running total down the sorted list, and a token is cut once the tokens above it already cover p."
hint: "`cum - sorted_probs` is everything ranked above a token. When should that token be cut?"
```

```card
type: calc
id: nucleus_size
eyebrow: Do the math
title: With p = 0.9, how many tokens does top-p keep?
given:
  - { label: Sorted probabilities, value: "0.50, 0.20, 0.15, 0.08, 0.04, 0.03" }
  - { label: p, value: "0.9" }
unit: tokens
answer: { min: 4, max: 4 }
placeholder: "0"
why_ok: "Running totals: 0.50, 0.70, 0.85, 0.93. The 4th token takes the total past 0.9, so it stays, and the last two go."
why_wrong: "Add the probabilities from the top until the total reaches 0.9. The token that crosses 0.9 is kept."
hint: Keep adding from the top. Stop after the token that takes the total past 0.9.
```

```card
type: quiz
ref: confident_model
```

```card
type: quiz
ref: why_seed
```

```card
type: quiz
ref: temperature_first
```

## Build steps

### Write top-k

```challenge
id: top_k
file: sampling.py
minutes: 20
start:
  code: |+
    import torch
    import torch.nn.functional as F

    # The tiny model from 2.1: a bigram table counted from one line of text plus one attention head.
    # ids [T] in, logits [T, V] out.
    text = "the cat sat on the mat. the dog sat on the log. a cat and a dog ran to the mat. "
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    encode = lambda s: torch.tensor([stoi[c] for c in s])
    decode = lambda ids: "".join(chars[i] for i in ids.tolist())


    class TinyLM:
        def __init__(self, seed=0):
            g = torch.Generator().manual_seed(seed)
            V, C, CTX = len(chars), 16, 64
            self.C = C
            counts = torch.ones(V, V) * 0.1
            for a, b in zip(text, text[1:]):
                counts[stoi[a], stoi[b]] += 1
            self.bigram = (counts / counts.sum(dim=1, keepdim=True)).log()
            self.wte = torch.randn(V, C, generator=g)
            self.wpe = torch.randn(CTX, C, generator=g) * 0.5
            self.Wq = torch.randn(C, C, generator=g) / C**0.5
            self.Wk = torch.randn(C, C, generator=g) / C**0.5
            self.Wv = torch.randn(C, C, generator=g) / C**0.5
            self.Wo = torch.randn(C, V, generator=g) * 0.15
            self.positions = 0  # how many positions the model has processed, in total

        def __call__(self, idx):
            T = idx.shape[0]
            self.positions += T
            x = self.wte[idx] + self.wpe[:T]
            q, k, v = x @ self.Wq, x @ self.Wk, x @ self.Wv
            att = (q @ k.T) / self.C**0.5
            att = att.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf"))
            h = F.softmax(att, dim=-1) @ v
            return self.bigram[idx] + h @ self.Wo


    model = TinyLM()

    prompt = encode("the ")


    def greedy(logits):
        return logits.argmax()


    def sample(logits, temperature=1.0, generator=None, keep=None):
        # 2.1's temperature sampling, plus an optional filter that cuts unlikely tokens first
        logits = logits / temperature
        if keep is not None:
            logits = keep(logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1, generator=generator)


    def generate(model, idx, max_new_tokens, pick=greedy):
        for _ in range(max_new_tokens):
            next_id = pick(model(idx)[-1])
            idx = torch.cat([idx, next_id.view(1)])
        return idx

setup:
  - say: This is 2.1's tiny model and loop. sample now takes an optional keep function, which cuts unlikely tokens before softmax.
    pause: 3.0
  - say: "Top-k first: keep the k largest logits, and set everything else to minus infinity so softmax gives it exactly zero."
    pause: 3.2
  - say: "Here's the stub. Your turn: find the k-th largest logit, then cut everything below it."
    type_file: top_k/stub.py
    pause: 1.2
tests: top_k/tests.py
solution: top_k/solution.py
alternatives:
  - top_k/alternatives/sort.py
  - top_k/alternatives/where.py
hints:
  logic: |
    find the k largest logits
      → kth = the smallest of those   (the cutoff)
    for each logit in the row:
      below kth?
        yes → replace it with -inf   (softmax gives it 0)
        no  → keep it
    return the row, still one logit per token
  tools: |
    find the cutoff
      t.topk(n)          the n largest values and their positions
      .values            just the values, sorted largest first
      t[-1]              the last item of t
    cut what is below it
      t < x              True where a value is below x
      t.masked_fill(mask, v)   replaces where mask is True
  nudge: Which single number decides the cut? Every logit at or above it stays, and every one below it goes.
  concept: "`logits.topk(k)` returns the k largest as `(values, indices)`, sorted from the top, so the cutoff is the last of the values. Then `masked_fill(mask, value)` replaces every position where `mask` is True. Use `-inf`, because softmax turns it into exactly 0."
  example: |
    Trace it on a tiny row, `logits = [0.5, 3.0, -1.0, 2.0, 1.0]` with `k = 2`:

    - The two largest values are `3.0` and `2.0`. The smaller of them is the cutoff, so `kth = 2.0`.
    - Below the cutoff: `0.5`, `-1.0` and `1.0` (positions 0, 2 and 4). The value `2.0` itself is not below it, so it stays.
    - After the cut: `[-inf, 3.0, -inf, 2.0, -inf]`. Still 5 entries, one per token.
    - Softmax of that: `[0, 0.731, 0, 0.269, 0]`. The cut tokens get exactly 0.
  skeleton: |
    kth = logits.____(k).______[___]
    return logits.masked_fill(logits ___ kth, float("____"))
  solution: |
    kth = logits.topk(k).values[-1]
    return logits.masked_fill(logits < kth, float("-inf"))
on_solve: "Three tokens left, and the probabilities show it: the cut tokens get exactly 0. Step 2 replaces the fixed k with a probability budget."
mistakes:
  - { file: top_k/mistakes/placeholders.py, fails: same_shape, says: "still returns `...`" }
  - { file: top_k/mistakes/return_placeholder.py, fails: same_shape, says: "still returns `...`" }
  - { file: top_k/mistakes/values_only.py, fails: same_shape, says: "returned only the 3 largest values" }
  - { file: top_k/mistakes/indices.py, fails: keeps_top_k, says: "should be a logit *value*" }
  - { file: top_k/mistakes/kth_is_max.py, fails: keeps_top_k, says: "kept only 1 token, the largest" }
  - { file: top_k/mistakes/le.py, fails: keeps_top_k, says: "the 3rd largest was cut too" }
  - { file: top_k/mistakes/flipped.py, fails: keeps_top_k, says: "you kept the *smallest* logits" }
  - { file: top_k/mistakes/zero_fill.py, fails: cut_to_zero_prob, says: "0 is an ordinary logit" }
  - { file: top_k/mistakes/no_main.py, fails: script_runs, says: "are missing" }
```

### Write top-p

```challenge
id: top_p
file: sampling.py
minutes: 30
start: { from: top_k }
setup:
  - say: Top-k keeps the same number of tokens whether the model is sure or torn. Top-p adapts. Rank the tokens, add their probabilities from the top, and stop once the total reaches p.
    type_file: top_p/stub.py
    pause: 2.4
  - say: "The sorting and the way back to vocabulary order are done. Your turn: the running total, and the rule for which tokens get cut."
    pause: 1.2
tests: top_p/tests.py
solution: top_p/solution.py
alternatives:
  - top_p/alternatives/shifted.py
  - top_p/alternatives/torch_cumsum.py
hints:
  logic: |
    sorted_probs = probabilities, largest first
      → cum = running total of sorted_probs
    for each token, in sorted order:
      the mass above it already reaches p?
        yes → cut it
        no  → keep it   (the token that crosses p stays)
    put the cut flags back in vocabulary order
      → cut logits become -inf → return
  tools: |
    running total of the sorted probabilities
      t.cumsum(dim)      running total along dim
    mass above each token, then the cut test
      plain arithmetic and a comparison against p
  nudge: Walk down the sorted list with a running total. For each token, look at the total of the tokens *above* it. When is that token no longer needed?
  concept: "`cumsum` gives the running total, including each token itself. Subtract the token's own probability and you get the mass of everything ranked above it. If that already reaches p, the token is cut. The token that crosses p is kept, so the most likely token always survives."
  example: |
    Trace it on `logits = [1.0, 3.0, 0.0, 2.0]` with `p = 0.8`:

    - Probabilities: `[0.087, 0.644, 0.032, 0.237]`.
    - Sorted: `[0.644, 0.237, 0.087, 0.032]`, from tokens `[1, 3, 0, 2]`.
    - `cum`: `[0.644, 0.881, 0.968, 1.0]`.
    - Mass above each token (`cum` minus its own): `[0, 0.644, 0.881, 0.968]`.
    - Cut where the mass above already reaches 0.8: `[False, False, True, True]`. Token 3 is the one that pushes the total past 0.8, and it is kept.
    - Back in vocabulary order: tokens 0 and 2 are cut. The result is `[-inf, 3.0, -inf, 2.0]`.

    With `p = 0.6` the second token's mass above (0.644) already covers it, so only token 1 survives.
  skeleton: |
    cum = sorted_probs.______(dim=-1)
    cut_sorted = cum - ____________ __ p
  solution: |
    cum = sorted_probs.cumsum(dim=-1)
    cut_sorted = cum - sorted_probs >= p
on_solve: "p = 0.5 keeps 1 token, p = 0.9 keeps 3, and p = 0.99 keeps all 6. The size follows the probabilities. Step 3 measures what each strategy does to the text."
mistakes:
  - { file: top_p/mistakes/placeholders.py, fails: keeps_something, says: "is still `...`" }
  - { file: top_p/mistakes/cut_placeholder.py, fails: keeps_something, says: "`cut_sorted` is still `...`" }
  - { file: top_p/mistakes/cum_gt_p.py, fails: keeps_something, says: "The token whose running total crosses p got cut" }
  - { file: top_p/mistakes/own_prob.py, fails: keeps_something, says: "cuts each token by its own probability" }
  - { file: top_p/mistakes/flipped.py, fails: keeps_something, says: "You kept the unlikely tail" }
  - { file: top_p/mistakes/no_cumsum.py, fails: nucleus, says: "kept every token" }
  - { file: top_p/mistakes/unsorted_cum.py, fails: nucleus, says: "in vocabulary order" }
  - { file: top_p/mistakes/no_main.py, fails: script_runs, says: "is missing" }
```

### Measure repetition

```challenge
id: distinct
file: sampling.py
minutes: 20
start: { from: top_p }
setup:
  - say: "How repetitive is a text? distinct-2 counts the pairs of neighbouring tokens, and asks what fraction of them are unique. \"the the the\" scores low."
    type_file: distinct/stub.py
    pause: 2.4
  - say: "Your turn: write distinct_2. The loop then runs four strategies from the same seed and scores each one."
    pause: 1.2
tests: distinct/tests.py
solution: distinct/solution.py
alternatives:
  - distinct/alternatives/comprehension.py
  - distinct/alternatives/set_first.py
hints:
  logic: |
    pairs = each id together with the id right after it
      (n ids give n − 1 pairs)
    count how many of those pairs are different from each other
      → return different pairs ÷ all pairs
    all different → 1.0, more repeats → closer to 0
  tools: |
    pair each id with the next one
      zip(a, b)          pairs up items of a and b, as tuples
      x[n:]              everything from index n on
      list(it)           collects an iterator into a list
    count the different pairs
      set(items)         one copy of each distinct item
      len(c)             how many items in c
    divide
      plain arithmetic: /
  nudge: How many neighbouring pairs does a list of n ids have? And how do you count the different ones?
  concept: "`zip(ids, ids[1:])` pairs each id with the next one, as tuples. A `set` keeps one copy of each distinct pair. distinct-2 is the number of distinct pairs divided by the number of pairs, n − 1."
  example: |
    Trace it on `ids = [4, 4, 7, 4, 4, 7, 9]`:

    - Pairs: `(4, 4)`, `(4, 7)`, `(7, 4)`, `(4, 4)`, `(4, 7)`, `(7, 9)`. That's 6 pairs from 7 ids.
    - Different ones: `(4, 4)`, `(4, 7)`, `(7, 4)`, `(7, 9)`. That's 4.
    - Result: 4 / 6 ≈ `0.667`.

    Check it on `[1, 2, 3, 4]`: 3 pairs, all different, so the score is `1.0`.
  skeleton: |
    pairs = list(___(ids, ids[__:]))
    return len(___(pairs)) / len(_____)
  solution: |
    pairs = list(zip(ids, ids[1:]))
    return len(set(pairs)) / len(pairs)
on_solve: "Greedy scores 0.08: \"the the the\". T = 1.0 scores highest, 0.62, but read it, because much of that is noise. Top-k and top-p sit in between with more real words. distinct-2 alone can't tell variety from garbage, which is why the metric step asks you to read samples too."
mistakes:
  - { file: distinct/mistakes/placeholders.py, fails: fraction, says: "still returns `...`" }
  - { file: distinct/mistakes/return_placeholder.py, fails: fraction, says: "still returns `...`" }
  - { file: distinct/mistakes/lists.py, fails: fraction, says: "lists can't go in a set" }
  - { file: distinct/mistakes/distinct_1.py, fails: fraction, says: "That's distinct-1" }
  - { file: distinct/mistakes/per_token.py, fails: fraction, says: "n − 1 neighbouring pairs" }
  - { file: distinct/mistakes/count.py, fails: fraction, says: "the number of unique pairs" }
  - { file: distinct/mistakes/inverted.py, fails: fraction, says: "upside down" }
  - { file: distinct/mistakes/skip_one.py, fails: fraction, says: "Your pairs skip a token" }
  - { file: distinct/mistakes/no_main.py, fails: script_runs, says: "is missing" }
```

### Compare strategies on your 30M model

```replay
id: sampling_30m
intro: On your GPU, sample 100 stories per strategy from your 30M model, with the KV cache from 2.2. Watch a recording first.
log: replays/sampling-30m.jsonl
speed: 40
recorded:
  hardware: RTX 3090
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: distinct-2 by strategy
  flag_above: 0.7
  x_labels: [greedy, T=1.5]
  caption: Greedy repeats one story 100 times. T = 1.5 scores highest and reads worst. Top-p 0.9 keeps the variety without the noise.
summary:
  - { label: distinct-2 (top-p 0.9), value: "0.41" }
  - { label: Sampling overhead, value: 86 µs/token }
  - { label: Share of a step, value: about 4% }
run_yourself: |
  # on your machine or Colab, from the course repo
  python sample.py --ckpt runs/tinystories-30m/best.pt \
    --prompt "Once upon a time" --samples 100 --new-tokens 200 --seed 0 --compare
```

### Record your metrics

```metrics
id: record
intro: distinct-2 and the sampling overhead for top-p 0.9, from your own run. Read a few samples before you trust the number.
```

## Hints

- On the GPU, filter a whole batch at once: `logits` is `[B, V]`, `topk` and `sort` work along `dim=-1`, and the cutoff needs `keepdim`-style indexing (`topk(k).values[:, -1:]`) so it broadcasts.
- Top-p sorts the whole vocabulary every token. With 128k-token vocabularies that's measurable, so some libraries apply top-k first and run top-p on the survivors.
- Use `torch.Generator(device="cuda")` on the GPU. A CPU generator can't draw for CUDA tensors.
- Repetition penalties and min-p are other common knobs. They're one line each once top-k and top-p make sense.

## Done when

- [ ] top_k keeps exactly the k largest logits {#top_k auto=challenge:top_k}
- [ ] top_p keeps the smallest top set covering p {#top_p auto=challenge:top_p}
- [ ] distinct-2 and sampling overhead recorded for top-p on your 30M model {#metrics_recorded auto=metrics}
