---
id: "0.3"
title: makemore basics
summary: A character-level name generator two ways, by counting pairs and by training a one-layer network, and the loss that compares them.
phase: "0"
hours: 3
type: build
status: ready
protected: false
cut: null
prerequisites:
  - "0.2"
papers:
  - title: A Neural Probabilistic Language Model
    authors: Bengio et al.
    year: 2003
quality_metric:
  id: val_loss
  label: Bigram/MLP validation loss
  unit: nats
  note: On your machine, all of names.txt. Train on a shuffled 90% of the names and measure on the other 10%. A bigram lands near 2.45; the MLP from Bengio et al. (makemore part 2) gets near 2.2.
  inputs:
    - { id: val_loss, label: Validation loss (nats), hint: "Held-out names, never trained on" }
  compute: value
  reference: 2.45
  pass: { max: 2.5 }
systems_metric:
  id: train_seconds
  label: Training time
  unit: s
  note: 100 steps of the neural bigram on all 228,146 pairs. The reference is an estimate for a recent GPU; a laptop CPU takes a few seconds.
  inputs:
    - { id: train_seconds, label: Seconds for 100 steps, hint: "time.perf_counter() around the loop" }
  compute: value
  reference: 1.5
  pass: { max: 60 }
quiz:
  - id: zero_count
    q: "Without the +1, the pair \"jq\" never shows up in training. What's the model's loss on a name containing it?"
    answer: 0
    options:
      - text: Infinite
        why: "P[j, q] is exactly 0, and −log 0 is infinite, so one name sends the average to infinity. Adding 1 to every count (smoothing) gives every pair a small chance."
      - text: High, but finite
        why: Only if the probability is above 0. A pair with a count of 0 gets probability exactly 0 before smoothing.
      - text: Zero, because the pair was never seen
        why: "Unseen pairs aren't skipped. The model still has to score them, and it said they were impossible."
  - id: nn_vs_counts
    q: After enough gradient descent, how does the one-layer neural bigram compare with the count table?
    answer: 0
    options:
      - text: About the same loss. Both are bigram models, and W ends up close to the log of the counts
        why: "One-hot @ W picks row x of W, so W is a table of log-counts. Gradient descent finds roughly what counting gives directly. The payoff comes when the input is more than one character, which counting can't scale to."
      - text: Much lower, because a neural network beats counting
        why: "It sees exactly the same information, one previous character. With one character of context, counting is already the best you can do."
      - text: Higher, because gradient descent can't reach the counts
        why: It gets very close. The gap you see after 100 steps closes with more steps, and W.pow(2) regularisation plays the role of +1 smoothing.
---

## Goal

Build makemore's first model, a **bigram**: predict each character from the one before it. Count every pair in a set of names, turn the counts into probabilities, sample new names, and measure the model with the average negative log-likelihood. Then get the same model a second way: one layer of weights trained by gradient descent, which is the form that scales to transformers.

## Learn

```card
type: concept
id: bigram_idea
eyebrow: The model
title: Guess the next letter from the letter before it.
body: |
  Put a "." at the start and end of every name. Then `.emma.` gives five pairs of neighbours: `.e`, `em`, `mm`, `ma` and `a.`.

  Count every pair in all the names into a 27 by 27 table. Row "e" holds how often each letter came right after "e". Divide each row by its total, and the counts become chances.

  To make up a name, start at ".", pick the next letter at random using that row's chances, and repeat until you pick "." again. To score the model, use the loss from 0.1: the average surprise at each right answer.
```

```card
type: predict
id: predict_smoothing
eyebrow: Predict the output
title: What prints?
code: |
  N = torch.tensor([[0, 3, 1],
                    [2, 0, 2],
                    [4, 0, 0]])
  P = (N + 1).float()
  P = P / P.sum(1, keepdim=True)
  print(P[2].tolist())
options:
  - text: "[1.0, 0.0, 0.0]"
    why: That's row 2 without the +1. Smoothing adds one to every count first, so no pair is impossible.
  - text: "[0.714, 0.143, 0.143]"
    correct: true
    why: "Row 2 becomes [5, 1, 1] after the +1, which sums to 7: 5/7, 1/7, 1/7."
  - text: "[5.0, 1.0, 1.0]"
    why: That's the smoothed counts before dividing. P divides each row by its own total.
  - text: "[0.556, 0.111, 0.111]"
    why: "That divides the row by 9 instead of its own total. Each row is divided by its own total, 7."
hint: Add 1 to each of 4, 0, 0, then divide by their new total.
```

```card
type: calc
id: name_loss
eyebrow: Do the math
title: What's the model's loss on the name "ab"?
given:
  - { label: "P(a | .)", value: "0.2" }
  - { label: "P(b | a)", value: "0.05" }
  - { label: "P(. | b)", value: "0.5" }
unit: nats
answer: { min: 1.75, max: 1.78 }
placeholder: "0.00"
why_ok: "−(ln 0.2 + ln 0.05 + ln 0.5) / 3 = (1.609 + 2.996 + 0.693) / 3 ≈ 1.766. The unlikely middle pair contributes most of it."
why_wrong: "Three pairs: \".a\", \"ab\" and \"b.\". Take −ln of each probability, then average the three."
hint: "−ln 0.5 ≈ 0.69, −ln 0.2 ≈ 1.61, −ln 0.05 ≈ 3.00."
```

```card
type: predict
id: predict_one_hot
eyebrow: Predict the output
title: What does a one-hot vector times a matrix give?
code: |
  W = torch.arange(9.0).reshape(3, 3)
  x = F.one_hot(torch.tensor([2]), num_classes=3).float()
  print(x @ W)
options:
  - text: "tensor([[6., 7., 8.]])"
    correct: true
    why: "x is [0, 0, 1], so x @ W adds up 0 × row 0, 0 × row 1 and 1 × row 2. Multiplying by a one-hot vector just picks a row."
  - text: "tensor([[2., 5., 8.]])"
    why: "That's column 2. A one-hot *row* vector on the left picks a row of W."
  - text: "tensor([[21.]])"
    why: "x has shape [1, 3] and W [3, 3], so the result is [1, 3], not a single number."
hint: "Write x out as [0, 0, 1] and do the row-times-matrix product by hand."
```

```card
type: fill_blank
id: softmax_loss_blank
eyebrow: Fill the blanks
title: Logits to probabilities to loss.
code: |
  logits = xenc @ W
  counts = logits.[[0]]()
  probs = counts / counts.sum(1, keepdim=True)
  loss = -probs[torch.arange(n), ys].[[1]]().mean()
tokens: [exp, log, abs, sqrt]
answer: [exp, log]
tips:
  abs: "abs makes values positive, but it loses the ordering: −3 and 3 would count the same. exp keeps bigger logits bigger."
  sqrt: sqrt of a negative logit is NaN. You need something defined for every number and always positive.
why_ok: "exp turns log-counts into positive counts, normalising turns counts into probabilities, and −log of the right one is the loss. The first two lines together are softmax."
why_wrong: "Logits are log-counts, so undo the log to get counts. The loss is built from the log of a probability."
hint: The logits are read as log-counts. Which function undoes a log?
```

```card
type: quiz
ref: zero_count
```

```card
type: quiz
ref: nn_vs_counts
```

## Build steps

### Count bigrams

```challenge
id: count_bigrams
file: bigram.py
minutes: 25
start:
  code: |+
    import torch

    # 81 names, enough to see the idea. On your machine you'll train on all 32,033 in names.txt.
    words = """
    emma olivia ava isabella sophia charlotte mia amelia harper evelyn abigail emily elizabeth mila
    ella avery sofia camila aria scarlett victoria madison luna grace chloe penelope layla riley zoey
    nora lily eleanor hannah lillian addison aubrey ellie stella natalie zoe liam noah oliver elijah
    william james benjamin lucas henry alexander mason michael ethan daniel jacob logan jackson levi
    sebastian mateo jack owen theodore aiden samuel joseph john david wyatt matthew luke asher carter
    julian grayson leo jayden gabriel isaac lincoln anthony
    """.split()

    chars = "abcdefghijklmnopqrstuvwxyz"
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}

setup:
  - say: "Here are 81 names and the vocabulary from 0.0: \".\" is 0, a to z are 1 to 26."
    pause: 2.4
  - say: "The model is a 27 by 27 table of counts. Row is the current character, column is the next one. Your turn: fill in the pairs and the count."
    type_file: count_bigrams/stub.py
    pause: 1.2
tests: count_bigrams/tests.py
solution: count_bigrams/solution.py
alternatives:
  - count_bigrams/alternatives/string.py
  - count_bigrams/alternatives/two_brackets.py
hints:
  nudge: "Every name needs a start marker and an end marker. Then each pair adds one to the cell at (row of the first character, column of the second)."
  concept: "`[\".\"] + list(w) + [\".\"]` gives the characters with a \".\" at each end, and `zip(chs, chs[1:])` walks every neighbouring pair. `stoi` turns each character into its row or column number, and `N[i, j] += 1` adds one to a cell."
  skeleton: |
    chs = [___] + list(_) + [___]
    N[stoi[____], stoi[____]] += _
  solution: |
    chs = ["."] + list(w) + ["."]
    N[stoi[ch1], stoi[ch2]] += 1
on_solve: "542 pairs from 81 names. Row \".\" is the distribution of first letters: try `print(N[0])`. Step 2 turns the counts into probabilities and generates names."
mistakes:
  - { file: count_bigrams/mistakes/placeholders.py, fails: returns_counts, says: "`chs` is still `...`" }
  - { file: count_bigrams/mistakes/add_placeholder.py, fails: returns_counts, says: "N is all zeros" }
  - { file: count_bigrams/mistakes/no_dots.py, fails: start_and_end, says: "nothing ever follows the start marker" }
  - { file: count_bigrams/mistakes/start_only.py, fails: start_and_end, says: "no name ever ends" }
  - { file: count_bigrams/mistakes/assign.py, fails: start_and_end, says: "`=` stores a 1 each time" }
  - { file: count_bigrams/mistakes/transposed.py, fails: right_cells, says: "N is transposed" }
```

### Sample names

```challenge
id: sample_name
file: bigram.py
minutes: 25
start: { from: count_bigrams }
setup:
  - say: "First, counts to probabilities: add one to every count so no pair is impossible, then divide each row by its total. That's 0.0's normalize_rows."
    type: "\n# Counts to probabilities. The +1 gives every pair a small chance, even ones never seen.\nP = (N + 1).float()\nP = P / P.sum(1, keepdim=True)\n\n"
    pause: 3.0
  - say: "To generate, start at \".\", draw the next character from that row, move to the row of what you drew, and stop when you draw \".\" again. Your turn: write the draw."
    type_file: sample_name/stub.py
    pause: 1.2
tests: sample_name/tests.py
solution: sample_name/solution.py
alternatives:
  - sample_name/alternatives/method.py
  - sample_name/alternatives/int_cast.py
hints:
  nudge: Row ix of P is the distribution of what comes after character ix. Draw one index from it at random, in proportion to the probabilities, using the generator g so a seed repeats the same names.
  concept: "`torch.multinomial(probs, num_samples=1, replacement=True, generator=g)` draws an index from a 1-D tensor of weights and returns it as a one-element tensor. `.item()` turns that into a plain int, which is what `itos` and the `== 0` check need."
  skeleton: |
    ix = torch.multinomial(P[__], num_samples=_, replacement=True, generator=_).____()
  solution: |
    ix = torch.multinomial(P[ix], num_samples=1, replacement=True, generator=g).item()
on_solve: "Mostly gibberish, and that's the point: a bigram only ever looks one character back, and 81 names leave most rows mostly smoothing. With all 32,033 names they get a little more name-like. Step 3 puts a number on how good the model is."
mistakes:
  - { file: sample_name/mistakes/placeholder.py, fails: returns_name, says: "`ix` is still `...`" }
  - { file: sample_name/mistakes/no_item.py, fails: returns_name, says: "keyed by plain Python ints" }
  - { file: sample_name/mistakes/whole_matrix.py, fails: returns_name, says: "Which row of P should each draw use?" }
  - { file: sample_name/mistakes/row_zero.py, fails: returns_name, says: "the same letter over and over" }
  - { file: sample_name/mistakes/argmax.py, fails: follows_probs, says: "Taking the most likely letter gives the same name every time" }
  - { file: sample_name/mistakes/no_generator.py, fails: uses_generator, says: "Pass the generator into the draw" }
```

### Score the model

```challenge
id: bigram_nll
file: bigram.py
minutes: 20
start: { from: sample_name }
setup:
  - say: "How good is P? Walk every pair in the data, take the log of the probability P gave to what actually came next, and average. Your turn: fill in nll."
    type_file: bigram_nll/stub.py
    pause: 1.6
  - say: "The last line prints 0.1's uniform baseline for comparison."
    pause: 1.0
tests: bigram_nll/tests.py
solution: bigram_nll/solution.py
alternatives:
  - bigram_nll/alternatives/method_log.py
hints:
  nudge: "P's row is the character you came from and its column the one that came next. Sum the logs as you go, then turn the sum into an average loss."
  concept: "`P[stoi[ch1], stoi[ch2]]` is the probability the model gave this pair. Its log is at most 0. Averaging the logs and flipping the sign gives the negative log-likelihood: lower is better, and 0 would mean every pair was predicted with certainty."
  skeleton: |
    logprob = torch.___(P[stoi[___], stoi[___]])
    return -______________ / _
  solution: |
    logprob = torch.log(P[stoi[ch1], stoi[ch2]])
    return -log_likelihood / n
on_solve: "About 2.47, against the uniform model's 3.30. Counting pairs bought you 0.83 nats. Step 4 trains a network to the same place by gradient descent."
mistakes:
  - { file: bigram_nll/mistakes/placeholders.py, fails: returns_number, says: "`logprob` is still `...`" }
  - { file: bigram_nll/mistakes/return_placeholder.py, fails: returns_number, says: "`nll` still returns `...`" }
  - { file: bigram_nll/mistakes/no_minus.py, fails: matches_reference, says: "The sign is flipped" }
  - { file: bigram_nll/mistakes/no_average.py, fails: matches_reference, says: "That's the total over all 10 pairs" }
  - { file: bigram_nll/mistakes/over_27.py, fails: matches_reference, says: "You divided by 27" }
  - { file: bigram_nll/mistakes/no_log.py, fails: matches_reference, says: "You averaged the probabilities themselves" }
  - { file: bigram_nll/mistakes/swapped.py, fails: matches_reference, says: "You read P[ch2, ch1]" }
```

### Train the neural bigram

```challenge
id: neural_bigram
file: neural.py
minutes: 40
start:
  code: |+
    import torch
    import torch.nn.functional as F

    # 81 names, enough to see the idea. On your machine you'll train on all 32,033 in names.txt.
    words = """
    emma olivia ava isabella sophia charlotte mia amelia harper evelyn abigail emily elizabeth mila
    ella avery sofia camila aria scarlett victoria madison luna grace chloe penelope layla riley zoey
    nora lily eleanor hannah lillian addison aubrey ellie stella natalie zoe liam noah oliver elijah
    william james benjamin lucas henry alexander mason michael ethan daniel jacob logan jackson levi
    sebastian mateo jack owen theodore aiden samuel joseph john david wyatt matthew luke asher carter
    julian grayson leo jayden gabriel isaac lincoln anthony
    """.split()

    chars = "abcdefghijklmnopqrstuvwxyz"
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}

setup:
  - say: "Same names, new file. This time the model is a 27 by 27 weight matrix W, trained with autograd. First the training set: every pair as an input xs and a target ys."
    type_file: neural_bigram/pairs.py
    pause: 2.6
  - say: "One-hot encode each input and multiply by W. That picks one row of W per input: 27 logits, read as log-counts."
    pause: 3.0
  - say: "Your turn: softmax in forward, the loss in loss_fn, and the update at the bottom of the loop."
    type_file: neural_bigram/stub.py
    pause: 1.2
tests: neural_bigram/tests.py
solution: neural_bigram/solution.py
alternatives:
  - neural_bigram/alternatives/softmax.py
  - neural_bigram/alternatives/nll_loss.py
  - neural_bigram/alternatives/no_grad.py
hints:
  nudge: "Three pieces. Softmax: make the logits positive, then make each row sum to 1. Loss: one probability per row, the one at that row's target, then −log and average. Update: move W a step against its gradient."
  concept: "`logits.exp()` gives positive counts and dividing by `.sum(1, keepdim=True)` normalises each row. `probs[torch.arange(len(ys)), ys]` picks row i's probability for target ys[i], one per row. For the update, change `W.data` so autograd doesn't record it, and subtract: the gradient points uphill."
  skeleton: |
    counts = logits.___()
    probs = counts / counts.___(_, keepdim=____)
    return -probs[torch.arange(___(ys)), __].___().mean()
    W.data += -__ * W.____
  solution: |
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    return -probs[torch.arange(len(ys)), ys].log().mean()
    W.data += -50 * W.grad
on_solve: "2.08 after 100 steps. Counting without the +1 gives 2.06 on these names, so gradient descent is closing in on the count table: W is learning log-counts. The difference is that this version scales: phase 1 swaps the one-hot input for a whole context, and W for a transformer."
mistakes:
  - { file: neural_bigram/mistakes/placeholders.py, fails: softmax_rows, says: "`probs` is still `...`" }
  - { file: neural_bigram/mistakes/no_normalize.py, fails: softmax_rows, says: "What turns counts into probabilities?" }
  - { file: neural_bigram/mistakes/no_exp.py, fails: softmax_rows, says: "Some of your probabilities are negative" }
  - { file: neural_bigram/mistakes/grand_total.py, fails: softmax_rows, says: "The whole table sums to 1" }
  - { file: neural_bigram/mistakes/columns.py, fails: softmax_rows, says: "Your *columns* sum to 1" }
  - { file: neural_bigram/mistakes/table_index.py, fails: loss_value, says: "takes every row's probability for every target" }
  - { file: neural_bigram/mistakes/global_n.py, fails: loss_value, says: "the size of your script's training set" }
  - { file: neural_bigram/mistakes/loss_no_minus.py, fails: loss_value, says: "negative. Log-probabilities are negative" }
  - { file: neural_bigram/mistakes/no_update.py, fails: trains, says: "W never changed" }
  - { file: neural_bigram/mistakes/wrong_sign.py, fails: trains, says: "The loss went *up*" }
  - { file: neural_bigram/mistakes/leaf_inplace.py, fails: trains, says: "Update the numbers behind W without recording it" }
```

### Record your metrics

```metrics
id: record
intro: From your own run on all of names.txt. The validation loss is measured on names the model never trained on.
```

## Hints

- Get the data with `curl -O https://raw.githubusercontent.com/karpathy/makemore/master/names.txt`, then `words = open("names.txt").read().splitlines()`.
- For a validation split, shuffle the names with `random.seed(42); random.shuffle(words)` and hold out the last 10%. Count (or train) on the first 90% only.
- Counting 228,146 pairs with `N[a, b] += 1` takes a second or two in PyTorch. The neural bigram builds all the pairs once and trains on the whole set every step.
- Karpathy's video "The spelled-out intro to language modeling: building makemore" covers this lesson. His MLP follow-up (Bengio et al. 2003) is where the loss drops below 2.2.

## Done when

- [ ] Names sampled from your bigram table {#sample auto=challenge:sample_name}
- [ ] The neural bigram trains below 2.3 on the lesson's names {#neural auto=challenge:neural_bigram}
- [ ] On your machine, with all of names.txt and a held-out 10%, both versions land near 2.45 {#full_run}
- [ ] Validation loss and training time recorded {#metrics_recorded auto=metrics}
