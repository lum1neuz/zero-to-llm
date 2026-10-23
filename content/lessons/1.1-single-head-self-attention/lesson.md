---
id: "1.1"
title: Single-head self-attention
summary: Write scaled dot-product attention with a causal mask, wrap it in a learned head, and watch one head beat the bigram on Shakespeare.
phase: "1"
hours: 4
type: build
status: ready
protected: false
cut: null
prerequisites:
  - "0.4"
papers:
  - title: Attention Is All You Need
    authors: Vaswani et al.
    year: 2017
    url: https://arxiv.org/abs/1706.03762
quality_metric:
  id: val_loss
  label: Validation loss with attention
  unit: nats
  note: Bigram plus one head on Tiny Shakespeare. The bigram table alone reaches about 2.49.
  inputs:
    - { id: val_loss, label: "Val loss after 5,000 steps", hint: The last eval line of your run }
  compute: value
  reference: 2.40
  pass: { max: 2.47 }
systems_metric:
  id: attn_ms
  label: Attention forward time at T=256
  unit: ms
  note: One Head(32, 16) in the browser, timed by step 2's script.
  inputs:
    - { id: attn_ms, label: T=256 forward (ms), hint: "Printed by step 2: \"T=256 forward\"" }
  compute: value
  reference: 3.3
  pass: { max: 50 }
quiz:
  - id: why_scale
    q: A head has head_size 64 and you forget the 1/√d. What goes wrong?
    answer: 1
    options:
      - text: Nothing much; softmax normalises the scores anyway
        why: softmax normalises the *sum*, not the spread. Scores 8× too large make the biggest one win almost everything.
      - text: Scores grow about 8× too large, softmax goes nearly one-hot, and gradients through it vanish
        why: A dot product of 64 unit-variance terms has a standard deviation of √64 = 8. Dividing by √d brings it back to about 1, where softmax stays soft and trainable.
      - text: The dot products overflow float32
        why: Values around ±8, or even ±80, are nowhere near float32's limit of about 3 × 10³⁸. The problem is softmax's shape, not overflow.
      - text: Every token attends evenly to everything before it
        why: That's what happens when scores are too *small*. Unscaled scores are too big, so the weights get sharper, not flatter.
  - id: why_mask
    q: You train a GPT without the causal mask. What happens?
    answer: 2
    options:
      - text: Training is slower, but the model ends up the same
        why: The model ends up very different. It learns a shortcut that doesn't exist at generation time.
      - text: The output shapes change, so the code crashes
        why: Shapes are identical with or without the mask. That's why the bug is silent.
      - text: Each position can read the token it's asked to predict, so training loss collapses and generation is garbage
        why: Position t's target is token t + 1. Without the mask, position t can attend to it and copy it. When you generate, token t + 1 doesn't exist yet, so the shortcut is gone.
      - text: Nothing; the mask only matters when generating
        why: The mask matters most in training, where every position is predicted in parallel from one sequence that already contains the answers.
---

## Goal

Write **scaled dot-product attention** with a **causal mask**: score every query against every key, hide the future, and softmax each row into weights. Then wrap it in a learned `Head` with key, query and value layers, check that it never sees the future, and time it at T = 256. A recording shows what one head does to Tiny Shakespeare's validation loss.

## Learn

```card
type: concept
id: what_attention_does
eyebrow: The idea
title: Every token asks the tokens before it a question.
visual:
  kind: bars
  max: 1
  bars:
    - { label: The, value: 0.12, text: "0.12" }
    - { label: cat, value: 0.71, text: "0.71" }
    - { label: sat, value: 0.17, text: "0.17" }
body: |
  Each token makes three vectors. Its **query** says what it's looking for, its **key** says what it contains, and its **value** is what it passes on. The query of "sat" is dotted with the keys of "The", "cat" and "sat", softmax turns those scores into weights, and the output is the weighted average of their values. The bars show where "sat" looks: mostly at its subject. Tokens after "sat" are hidden, because when you generate they don't exist yet.
```

```card
type: predict
id: predict_average
eyebrow: Predict the output
title: All scores are zero. What does row 1 hold?
code: |
  T = 3
  scores = torch.zeros(T, T)
  tril = torch.tril(torch.ones(T, T))
  wei = scores.masked_fill(tril == 0, float("-inf")).softmax(dim=-1)
  print(wei[1].tolist())
options:
  - text: "[0.5, 0.5, 0.0]"
    correct: true
    why: Row 1 may see tokens 0 and 1. Their scores are equal, so softmax splits the weight evenly, and e^−inf gives token 2 exactly 0. With equal scores, attention is a plain average of the past.
  - text: "[0.3333, 0.3333, 0.3333]"
    why: That's an unmasked row. Token 2 comes after token 1, so its score is −inf, and softmax gives it nothing.
  - text: "[1.0, 1.0, 0.0]"
    why: That's row 1 of `tril` itself. softmax rescales the allowed entries so the row sums to 1.
  - text: "[0.0, 1.0, 0.0]"
    why: That would be a token attending only to itself. Equal scores mean equal weights on everything it's allowed to see.
hint: Row 1 may look at positions 0 and 1. What does softmax do with two equal scores and one −inf?
```

```card
type: fill_blank
id: mask_and_softmax
eyebrow: Fill the blanks
title: Hide the future, then normalise each row.
code: |
  scores = q @ k.transpose(-2, -1) / d ** 0.5
  masked = scores.masked_fill(tril == [[0]], [[1]])
  wei = masked.softmax(dim=[[2]])
tokens: ["0", "1", 'float("-inf")', "0.0", "-1", "-2"]
answer: ["0", 'float("-inf")', "-1"]
tips:
  "1": "`tril` is 1 where a token *may* look. Filling those would hide the past and keep the future."
  "0.0": softmax turns a score of 0 into e⁰ = 1, a real share of the weight. Hidden cells need a score that becomes exactly 0.
  "-2": That normalises down each column. Row t holds query t's weights over the keys, so each row must sum to 1.
why_ok: "Cells above the diagonal, where `tril` is 0, get −inf. softmax along each row turns −inf into 0 and shares 100% of the weight among the keys a token may see."
why_wrong: "Check each blank: which cells are the future, what score does softmax turn into exactly 0, and which dimension runs along a row?"
hint: "`tril` marks the cells a token may see with 1. You want to fill the other ones."
```

```card
type: calc
id: dot_product_size
eyebrow: Do the math
title: q and k each hold 64 random numbers with variance 1. How big is a typical q·k?
given:
  - { label: head_size d, value: "64" }
  - { label: Each entry, value: "mean 0, variance 1" }
  - { label: Answer with, value: the standard deviation of q·k }
unit: ""
answer: { min: 7.9, max: 8.1 }
placeholder: "0"
why_ok: q·k is a sum of 64 products, each with variance 1, so its variance is 64 and its standard deviation √64 = 8. Dividing by √d brings it back to about 1, where softmax stays soft.
why_wrong: Variances add. 64 independent products of variance 1 give variance 64. The standard deviation is the square root of that.
hint: Each product q_i·k_i has variance 1, and the variances of independent terms add up. Then take a square root.
```

```card
type: reorder
id: attention_order
eyebrow: Put it in order
title: Arrange one head's attention, from scores to output.
lines:
  - scores = q @ k.transpose(-2, -1) / d ** 0.5
  - masked = scores.masked_fill(tril == 0, float("-inf"))
  - wei = masked.softmax(dim=-1)
  - out = wei @ v
shuffle: [2, 0, 3, 1]
why_ok: "Score, hide, normalise, mix. The mask goes on before softmax so the hidden cells get exactly 0 and each row still sums to 1."
why_wrong: "Follow the names: `masked` needs `scores`, `wei` needs `masked`, and `out` needs `wei`."
hint: Each line can only use names an earlier line created.
```

```card
type: quiz
ref: why_scale
```

```card
type: quiz
ref: why_mask
```

## Build steps

### Write causal_weights

```challenge
id: causal_weights
file: attention.py
minutes: 35
start: { code: "import torch\n\n" }
setup:
  - say: Two small tensors stand in for a sequence. Five tokens, each with a query and a key of four numbers.
    type: "torch.manual_seed(42)\nq = torch.randn(5, 4)\nk = torch.randn(5, 4)\n\n"
    pause: 2.4
  - say: "The plan: score every query against every key, hide each token's future, and turn each row of scores into weights that sum to 1."
    pause: 3.6
  - say: "Here's the stub. Your turn: fill in scores, tril, masked and wei, then run the tests."
    type_file: causal_weights/stub.py
    pause: 1.2
tests: causal_weights/tests.py
solution: causal_weights/solution.py
alternatives:
  - causal_weights/alternatives/manual_softmax.py
  - causal_weights/alternatives/bool_mask.py
  - causal_weights/alternatives/functional.py
  - causal_weights/alternatives/where.py
  - causal_weights/alternatives/big_negative.py
hints:
  logic: |
    scores = every query against every key   (a T × T grid)
      → divide by √d
    tril = T × T grid, 1 on and below the diagonal, 0 above
    for each cell of the score grid:
      tril says 0?   (that token would peek ahead)
        yes → overwrite with a score softmax turns into 0
        no  → keep the score
    masked → softmax along each row   (weights sum to 1)
      → return wei
  tools: |
    score every query against every key
      a @ b               matrix multiply
      t.transpose(i, j)   swaps axes i and j of t
      x ** p              x to the power p (a root is a power)
    grid of ones, keep the lower triangle
      torch.ones(n, n)    n × n grid of ones
      torch.tril(t)       zeroes everything above the diagonal
    overwrite where the grid says 0
      t == x              True wherever t equals x
      t.masked_fill(m, v) puts v wherever mask m is True
    softmax along each row
      t.softmax(dim)      weights that sum to 1 along dim
  nudge: Three moves, score, hide, normalise. Which score does softmax turn into exactly zero weight?
  concept: "Row t of the score grid is query t dotted with every key, so k's last two dimensions have to swap before the `@`. Divide by √d to keep the scores near ±1. `torch.tril` of a grid of ones is 1 on and below the diagonal, the cells a token may see. Fill the rest with a score softmax turns into 0, then softmax along each row."
  example: |
    Trace it on three tokens with `d = 4`, so √`d` = 2:

    - `q = [[1,0,1,0], [0,2,0,0], [1,1,0,0]]` and `k = [[2,0,0,0], [0,1,0,1], [1,0,1,0]]`.
    - Raw scores, `q` against every `k`: `[[2,0,2], [0,2,0], [2,1,1]]`. Divided by 2: `scores = [[1,0,1], [0,1,0], [1,0.5,0.5]]`.
    - `tril` is `[[1,0,0], [1,1,0], [1,1,1]]`.
    - `masked` is `[[1,×,×], [0,1,×], [1,0.5,0.5]]`, where × is the score softmax turns into exactly 0.
    - `wei` is about `[[1,0,0], [0.269,0.731,0], [0.452,0.274,0.274]]`.

    Check: row 0 puts everything on token 0, every future cell is exactly 0, and every row sums to 1.
  skeleton: |
    scores = q @ k.__________(-2, -1) / d ** ___
    tril = torch.____(torch.ones(T, T))
    masked = scores.masked_fill(tril == _, float("____"))
    wei = masked.softmax(dim=__)
  solution: |
    scores = q @ k.transpose(-2, -1) / d ** 0.5
    tril = torch.tril(torch.ones(T, T))
    masked = scores.masked_fill(tril == 0, float("-inf"))
    wei = masked.softmax(dim=-1)
on_solve: "Solved. Look at the printed grid: row 0 is all on token 0, and every row sums to 1. Step 2 is unlocked: real q, k and v come from learned layers."
mistakes:
  - { file: causal_weights/mistakes/placeholders.py, fails: weights_shape, says: "`wei` is still a placeholder" }
  - { file: causal_weights/mistakes/scores_placeholder.py, fails: weights_shape, says: "`scores` is still `...`" }
  - { file: causal_weights/mistakes/no_transpose.py, fails: weights_shape, says: "the keys have to stand as columns" }
  - { file: causal_weights/mistakes/qt_k.py, fails: weights_shape, says: "one per pair of *features*" }
  - { file: causal_weights/mistakes/no_mask.py, fails: causal, says: "Nothing is hidden yet" }
  - { file: causal_weights/mistakes/zero_fill.py, fails: causal, says: "softmax turns a score of 0 into e⁰ = 1" }
  - { file: causal_weights/mistakes/triu.py, fails: causal, says: "zero *below* the diagonal" }
  - { file: causal_weights/mistakes/inverted.py, fails: causal, says: "hides each token's past and keeps its future" }
  - { file: causal_weights/mistakes/no_softmax.py, fails: causal, says: "softmax is missing" }
  - { file: causal_weights/mistakes/softmax_first.py, fails: rows_sum_to_one, says: "you masked after the softmax" }
  - { file: causal_weights/mistakes/dim0.py, fails: rows_sum_to_one, says: "Your *columns* sum to 1" }
  - { file: causal_weights/mistakes/unscaled.py, fails: matches_reference, says: "Your scores aren't scaled" }
  - { file: causal_weights/mistakes/div_d.py, fails: matches_reference, says: "You divided by d (16), not √d (4)" }
  - { file: causal_weights/mistakes/fixed_two.py, fails: matches_reference, says: "√d only when d = 4" }
  - { file: causal_weights/mistakes/k_at_q.py, fails: matches_reference, says: "Your grid is transposed" }
  - { file: causal_weights/mistakes/fixed_T.py, fails: batched, says: "Your mask is always 5 × 5" }
  - { file: causal_weights/mistakes/dot_T.py, fails: batched, says: "`.T` reverses *every* dimension" }
```

### Build a Head and time it

```challenge
id: head
file: attention.py
minutes: 35
start: { from: causal_weights }
setup:
  - say: Your weights work. In a model nobody hands you q, k and v. Each token's embedding is projected three ways by learned layers.
    type_file: head/stub.py
    pause: 2.4
  - say: "The script runs your head on a batch of 4 sequences, then times it at T = 256. That time is your systems metric. Your turn: fill in the layers and forward."
    pause: 1.2
tests: head/tests.py
solution: head/solution.py
alternatives:
  - head/alternatives/sdpa.py
  - head/alternatives/positional_bias.py
  - head/alternatives/matmul.py
hints:
  logic: |
    when the head is built:
      key, query, value = three learned layers
        each: n_embd features → head_size, no bias
    when x comes in:
      run x through each layer → k, q, v
      wei = causal_weights(q, k)
      mix the values (not x, k or q) by the weights
        → return the mix
  tools: |
    three learned layers
      nn.Linear(n_in, n_out, bias=b)   a learned linear map
    run x through each layer
      layer(x)            applies the layer to x
    mix the values by the weights
      a @ b               matrix multiply
  nudge: Key, query and value are each a learned map from n_embd numbers to head_size numbers. Which layer is a learned linear map? Then which of the three do the weights mix?
  concept: "`nn.Linear(n_embd, head_size, bias=False)` stores a [head_size, n_embd] weight. Call each layer on x to get k, q and v, each [B, T, head_size]. The keys and queries decide the weights; the weights then take a mix of the *values*."
  example: |
    Trace the shapes with the script's numbers, `n_embd = 32`, `head_size = 16`, and `x` of shape `[4, 8, 32]` (4 sequences, 8 tokens):

    - `k`, `q` and `v` are each `[4, 8, 16]`: every token went from 32 features to 16.
    - `wei` is `[4, 8, 8]`: for each sequence, how much each token listens to each other token.
    - `wei` times `v`: `[4, 8, 8]` times `[4, 8, 16]` gives `[4, 8, 16]`.

    Check: the script prints `out (4, 8, 16)`. A sequence and a token position in, a `head_size` vector out.
  skeleton: |
    self.key = nn.______(n_embd, _________, bias=_____)
    self.query = nn.______(n_embd, _________, bias=_____)
    self.value = nn.______(n_embd, _________, bias=_____)
    k = self.___(x)
    q = self._____(x)
    v = self._____(x)
    return wei @ _
  solution: |
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)
    return wei @ v
on_solve: "That's one head of self-attention. Copy the T=256 time into your metrics. Then change 256 to 512 and run again: the score grid grows 4× while the projections only double. That T × T grid is what the KV cache (2.2) and fused attention (2.5) are about."
mistakes:
  - { file: head/mistakes/placeholders.py, fails: projections, says: "`self.key` is still `...`" }
  - { file: head/mistakes/reversed_linear.py, fails: projections, says: "takes the input size first" }
  - { file: head/mistakes/with_bias.py, fails: projections, says: "has a bias" }
  - { file: head/mistakes/shared_layer.py, fails: projections, says: "the *same* layer" }
  - { file: head/mistakes/forward_placeholders.py, fails: output_shape, says: "is still `...`" }
  - { file: head/mistakes/return_wei.py, fails: output_shape, says: "the attention weights themselves" }
  - { file: head/mistakes/mix_x.py, fails: output_shape, says: "you mixed `x` itself" }
  - { file: head/mistakes/no_causal.py, fails: no_future, says: "earlier tokens can see later ones" }
  - { file: head/mistakes/swapped_qk.py, fails: matches_reference, says: "the names now lie" }
  - { file: head/mistakes/mix_keys.py, fails: matches_reference, says: "You mixed the keys" }
  - { file: head/mistakes/no_timing.py, fails: script_runs, says: "The timing line didn't print" }
```

### One head on Tiny Shakespeare

```replay
id: shakespeare_head
intro: "On your laptop, add your Head between the embeddings and the output layer of the bigram model and train for 5,000 steps. It takes a couple of minutes on a CPU. Here's what a healthy run looks like."
log: replays/shakespeare-head.jsonl
speed: 40
recorded:
  hardware: Laptop CPU (8 cores)
  illustrative: true
  note: Sample data written for the course, shaped like Karpathy's single-head result. Not a real recording. Replace it with a log of your own run.
chart:
  title: Validation loss every 500 steps
  flag_above: 2.49
  x_labels: [step 0, step 5000]
  caption: Red bars are still worse than the bigram table's 2.49. From about step 3000, one head of attention beats a model that only sees the previous character.
summary:
  - { label: Val loss, value: 2.49 → 2.40 }
  - { label: Parameters, value: "4,225 → 7,553" }
  - { label: Context, value: 1 → 8 characters }
run_yourself: |
  # on your laptop (CPU is fine), from the course repo
  python train_gpt.py --config shakespeare_head --max-iters 5000
```

### Record your metrics

```metrics
id: record
intro: Your val loss from your own run, and the T=256 time your head printed in the browser. They're compared with the reference run and the pass limit.
```

## Hints

- Swap only the last two dimensions of `k`: `k.transpose(-2, -1)` works for one sequence and for a batch. `.T` reverses every dimension, so it breaks on a batch.
- Build the mask from the input's own length, `T = q.shape[-2]`. In a model, T changes whenever you generate one more token.
- The mask goes on the scores, before softmax. Masking the weights after softmax leaves rows that no longer sum to 1.
- `F.scaled_dot_product_attention(q, k, v, is_causal=True)` does all of this in one fused call. Write it by hand once, then use the fused one in 1.2.

## Done when

- [ ] causal_weights works on a batch of any length {#causal_weights auto=challenge:causal_weights}
- [ ] Your Head never lets a token see the future {#head auto=challenge:head}
- [ ] Val loss and T=256 time recorded {#metrics_recorded auto=metrics}
