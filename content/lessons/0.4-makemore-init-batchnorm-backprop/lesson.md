---
id: "0.4"
title: makemore init, BatchNorm, manual backprop
summary: Why a fresh network starts out confidently wrong, the two fixes (Kaiming init and BatchNorm), and an MLP's backward pass written by hand and checked against autograd.
phase: "0"
hours: 2
type: build
status: ready
protected: false
cut: null
prerequisites:
  - "0.3"
papers:
  - title: Batch Normalization
    authors: Ioffe & Szegedy
    year: 2015
    url: https://arxiv.org/abs/1502.03167
  - title: Delving Deep into Rectifiers
    authors: He et al.
    year: 2015
    url: https://arxiv.org/abs/1502.01852
quality_metric:
  id: val_loss
  label: Validation loss after init fixes
  unit: nats
  note: The makemore MLP on all of names.txt, 200k steps, with the output layer scaled down and Kaiming init or BatchNorm. The reference comes from the illustrative replay.
  inputs:
    - { id: val_loss, label: Validation loss (nats), hint: Your best configuration }
  compute: value
  reference: 2.103
  pass: { max: 2.15 }
systems_metric:
  id: manual_grad_err
  label: Max manual-backprop error vs autograd
  unit: abs
  note: The largest maxdiff in step 3's printout. float32 rounding leaves a few times 1e-8.
  inputs:
    - { id: manual_grad_err, label: Largest maxdiff, hint: From step 3's printout }
  compute: value
  reference: 3.0e-8
  pass: { max: 1.0e-5 }
quiz:
  - id: bn_inference
    q: At inference you feed the model a single name. What does BatchNorm use for the mean and variance?
    answer: 0
    options:
      - text: Running averages of the batch statistics, collected during training
        why: "Each training step nudges `running_mean` and `running_var` towards that batch's statistics. `model.eval()` switches BatchNorm to use them, so one example gives the same output whatever else is in the batch."
      - text: The mean and variance of that one example
        why: "A batch of one has variance 0 in every feature, so there's nothing to normalize by. And the output would depend on what happens to share the batch."
      - text: Always mean 0 and variance 1
        why: "That skips normalizing altogether. The layer before BatchNorm can output anything, and the layers after it were trained on normalized inputs."
  - id: bias_before_bn
    q: Why can you drop the bias b1 from the layer right before a BatchNorm?
    answer: 0
    options:
      - text: BatchNorm subtracts each feature's mean, which removes any constant b1 added, and beta takes over its job
        why: "Adding b1 shifts a column and its mean by the same amount, so subtracting the mean cancels it exactly. b1's gradient is 0: a wasted parameter."
      - text: BatchNorm's gamma multiplies it by 0
        why: gamma scales the normalized values. The bias is gone before gamma is applied, because subtracting the mean removed it.
      - text: Biases don't matter in deep networks
        why: They do matter in general. This one is redundant only because BatchNorm cancels it and adds its own shift, beta.
---

## Goal

Make the makemore MLP train well from step 0. A fresh network with plain `randn` weights is confidently wrong, and its tanh units sit saturated where no gradient flows. Fix it two ways: **Kaiming init**, which scales each layer's weights by gain / √fan_in, and **BatchNorm**, which normalizes each layer's pre-activations over the batch. Then write the MLP's backward pass by hand, one gradient at a time, and check every one against autograd.

## Learn

```card
type: concept
id: confidently_wrong
eyebrow: Step 0
title: A new network shouldn't be sure of itself.
visual:
  kind: bars
  max: 28
  limit: { value: 3.3, label: "ln 27 ≈ 3.30" }
  bars:
    - { label: Plain randn, value: 27.9, text: "27.9" }
    - { label: Output × 0.01, value: 3.32, text: "3.32" }
body: |
  The bars show the loss at the very first step, before any training. A model that knows nothing, and admits it, scores about 3.30 (ln 27).

  With plain random starting weights, the scores the model gives each letter come out huge. Huge scores make it almost certain of random answers, so it starts at 27.9, far worse than knowing nothing. Training then wastes its first thousands of steps unlearning that false confidence.

  Making the last layer's starting weights tiny fixes it: the loss starts at 3.32. A similar problem inside the network, and two ways to fix it, are the rest of this lesson.
```

```card
type: calc
id: kaiming_std
eyebrow: Do the math
title: What std should W1 have?
given:
  - { label: Inputs per unit (fan_in), value: "30 (3 characters × 10-dim embedding)" }
  - { label: Activation, value: tanh }
  - { label: Kaiming std, value: "gain / √fan_in, tanh gain 5/3" }
unit: std
answer: { min: 0.30, max: 0.31 }
placeholder: "0.00"
why_ok: "(5/3) / √30 ≈ 1.667 / 5.477 ≈ 0.304. Karpathy's hand-tuned 0.2 was in the right neighbourhood."
why_wrong: "Divide the gain, 5/3 ≈ 1.667, by the square root of fan_in: √30 ≈ 5.48."
hint: "√30 is a bit less than 5.5."
```

```card
type: predict
id: predict_saturation
eyebrow: Predict the output
title: How much gradient gets back through tanh?
code: |
  hpreact = torch.tensor([0.5, 3.0, -8.0])
  h = torch.tanh(hpreact)
  print((1 - h**2).tolist())
options:
  - text: "about [0.79, 0.0099, 0.0000005]"
    correct: true
    why: "tanh(0.5) ≈ 0.46, so 1 − 0.21 ≈ 0.79. At 3 and −8, tanh is within a whisker of ±1, so almost nothing flows back. Those units are saturated, and their weights barely learn."
  - text: "about [0.75, −8.0, −63.0]"
    why: "That's 1 − hpreact², using the input. The formula takes tanh's output, h, which is always between −1 and 1."
  - text: "about [0.46, 0.995, −1.0]"
    why: That's tanh itself. Its slope is 1 − tanh², which shrinks to 0 as tanh approaches ±1.
hint: "tanh(3) ≈ 0.995. What's 1 − 0.995²?"
```

```card
type: fill_blank
id: batchnorm_blank
eyebrow: Fill the blanks
title: BatchNorm's statistics in training mode.
code: |
  # h is [batch, features]
  mean = h.mean([[0]], keepdim=True)
  var = h.var(0, keepdim=True, unbiased=[[1]])
tokens: ["0", "1", "True", "False"]
answer: ["0", "False"]
tips:
  "1": "dim=1 averages each example across its features. That's LayerNorm. BatchNorm averages each feature down the batch."
  "True": "unbiased=True divides by n − 1. BatchNorm normalizes with the batch's own variance, dividing by n, as nn.BatchNorm1d does in training."
why_ok: "Each feature gets its own mean and variance over the batch, dividing by n like nn.BatchNorm1d."
why_wrong: "BatchNorm's statistics run down the batch dimension, and it uses the plain variance of the batch, dividing by n."
hint: Which dimension is the batch? And does the batch need the n − 1 correction meant for estimating a whole population?
```

```card
type: reorder
id: backward_chain
eyebrow: Put it in order
title: Arrange the backward pass from the loss to W1.
lines:
  - dlogits = (F.softmax(logits, 1) - F.one_hot(Y, 27)) / n
  - dh = dlogits @ W2.T
  - dhpreact = (1 - h**2) * dh
  - dW1 = X.T @ dhpreact
shuffle: [2, 0, 3, 1]
why_ok: "Backprop runs the forward pass in reverse. Each gradient needs the one just above it: dh needs dlogits, dhpreact needs dh, dW1 needs dhpreact."
why_wrong: "Follow the names: every line uses the gradient the line before it created."
hint: Start at the loss. The forward pass ended with logits, so backward starts there.
```

```card
type: quiz
ref: bn_inference
```

```card
type: quiz
ref: bias_before_bn
```

## Build steps

### Kaiming init

```challenge
id: kaiming_init
file: init.py
minutes: 20
start: { code: "import torch\n\n" }
setup:
  - say: "Each output of a layer sums fan_in inputs times weights, so its spread grows like the square root of fan_in. Kaiming init divides the weights by that, and multiplies by a gain that makes up for what the activation squashes: 5/3 for tanh."
    pause: 4.4
  - say: "Your turn: write linear_weights. Below it, five tanh layers report their spread and how many units are saturated."
    type_file: kaiming_init/stub.py
    pause: 1.2
tests: kaiming_init/tests.py
solution: kaiming_init/solution.py
alternatives:
  - kaiming_init/alternatives/int_sizes.py
  - kaiming_init/alternatives/normal_.py
hints:
  logic: |
    draw a fan_in × fan_out table of standard normal numbers
      (use the generator g, so a seed repeats)
      → their spread (std) is 1
    wanted spread = gain ÷ √fan_in
      → multiply the whole table by it
      → return the weights
  tools: |
    draw the table of normal numbers
      torch.randn(shape, generator=g)   standard normal numbers
                                        shape is a tuple
    work out the scale
      plain arithmetic: * / and ** (power)
    scale the table
      t * x                             multiplies every value by x
  nudge: "Start from standard normal weights of the right shape, drawn with g. Their std is 1. What do you multiply them by to get gain / √fan_in?"
  concept: "`torch.randn((fan_in, fan_out), generator=g)` has std 1. Multiplying a tensor scales its std by the same factor, so multiply by `gain / fan_in**0.5`."
  example: |
    Trace it with `fan_in = 100`, `fan_out = 100`, `gain = 5/3`:

    - The draw has shape `[100, 100]` and std about `1`.
    - The target std is `(5/3) / √100 = 1.6667 / 10 ≈ 0.167`. Multiply the draw by that, and `W.std()` comes out near `0.167`.
    - Feed it `x` with std `1`. Each output sums 100 products, so its std is about `1 * 0.167 * √100 ≈ 1.67`. After `tanh`, the spread is about `0.76` and about 21% of units are saturated; deeper layers fall to about 6%.
    - Leave the scaling out and the std is `1 * √100 = 10` before `tanh`, so about 83% of the outputs sit at ±1.

    Another size: `fan_in = 30`, `fan_out = 200` gives a `[30, 200]` matrix with std `(5/3) / √30 ≈ 0.304`.
  skeleton: |
    return torch.randn((______, _______), generator=_) * ____ / ______**0.5
  solution: |
    return torch.randn((fan_in, fan_out), generator=g) * gain / fan_in**0.5
on_solve: "The spread settles near 0.66 and saturation falls layer by layer. Now try gain=1: the std shrinks at every layer. Try plain randn (gain = √100 = 10): almost every unit saturates. On a GPU, `torch.nn.init.kaiming_normal_(W, mode='fan_out', nonlinearity='tanh')` does the same for this (fan_in, fan_out) layout. PyTorch assumes nn.Linear's (out, in) layout, where the default mode='fan_in' is right."
mistakes:
  - { file: kaiming_init/mistakes/placeholder.py, fails: shape, says: "still returns `...`" }
  - { file: kaiming_init/mistakes/transposed.py, fails: shape, says: "the transpose" }
  - { file: kaiming_init/mistakes/no_scale.py, fails: kaiming_std, says: "Plain `randn` has std 1" }
  - { file: kaiming_init/mistakes/gain_only.py, fails: kaiming_std, says: "That's the gain alone" }
  - { file: kaiming_init/mistakes/no_gain.py, fails: kaiming_std, says: "with the gain left out" }
  - { file: kaiming_init/mistakes/no_sqrt.py, fails: kaiming_std, says: "You divided by fan_in itself" }
  - { file: kaiming_init/mistakes/fan_out.py, fails: kaiming_std, says: "You used fan_out" }
  - { file: kaiming_init/mistakes/uniform.py, fails: kaiming_std, says: "Every weight is positive" }
  - { file: kaiming_init/mistakes/no_generator.py, fails: uses_generator, says: "Pass `generator=g`" }
```

### BatchNorm by hand

```challenge
id: batchnorm
file: batchnorm.py
minutes: 25
start: { code: "import torch\n\n" }
setup:
  - say: "Kaiming init gets the scales right at step 0. BatchNorm keeps them right while training: normalize each feature over the batch, then let the network rescale with a learned gamma and beta."
    type_file: batchnorm/data.py
    pause: 3.2
  - say: "Your turn: write batchnorm. The bottom checks your output against PyTorch's nn.BatchNorm1d."
    type_file: batchnorm/stub.py
    pause: 1.2
tests: batchnorm/tests.py
solution: batchnorm/solution.py
alternatives:
  - batchnorm/alternatives/no_keepdim.py
  - batchnorm/alternatives/manual_var.py
hints:
  logic: |
    for each feature (column of h), looking down the batch:
      mean = the average
      variance = average squared distance from the mean
        (divide by n, not n − 1)
    centre: h − mean
      → divide by √(variance + eps)   (eps: never divide by 0)
      → multiply by gamma
      → add beta
      → return
  tools: |
    mean and variance of each column
      t.mean(dim, keepdim=True)         average along dim
      t.var(dim, keepdim=True, unbiased=b)
                                        variance along dim
                                        b=False divides by n
    centre, divide, scale, shift
      plain arithmetic: * / + -
      torch.sqrt(t)                     square root of every value
  nudge: "Statistics down the batch, one per feature. Then subtract, divide by a standard deviation that can't be 0, scale and shift."
  concept: "`h.mean(0, keepdim=True)` is a [1, features] row of column means. `.var(0, keepdim=True, unbiased=False)` divides by n, like nn.BatchNorm1d in training. The std is `torch.sqrt(var + eps)`: eps keeps a constant feature from dividing by 0. gamma multiplies, beta adds."
  example: |
    Trace it on 3 examples and 2 features, with `gamma = 1`, `beta = 0`:

    - `h = [[1, 10], [2, 14], [3, 6]]`.
    - Means per column: `[2, 10]`. Variances (divide by 3): `[0.667, 10.667]`. Stds: `[0.816, 3.266]`.
    - Centred: `[[-1, 0], [0, 4], [1, -4]]`.
    - Divided by the std: `[[-1.225, 0], [0, 1.225], [1.225, -1.225]]`.
    - Each column now has mean 0 and std 1, even though the second started out about four times wider.

    With `gamma = 2` and `beta = 1`, the first column becomes `2 * [-1.225, 0, 1.225] + 1 = [-1.449, 1, 3.449]`.

    Traps: dividing by `n − 1` gives a first column of `[-1, 0, 1]`, a bit too small. And a constant column like `[5, 5, 5]` has variance 0, which only works because `eps` is there. It comes out as all zeros.
  skeleton: |
    mean = h.____(_, keepdim=True)
    var = h.___(_, keepdim=True, unbiased=_____)
    return _____ * (h - ____) / torch.____(var + ___) + ____
  solution: |
    mean = h.mean(0, keepdim=True)
    var = h.var(0, keepdim=True, unbiased=False)
    return gamma * (h - mean) / torch.sqrt(var + eps) + beta
on_solve: "Matches nn.BatchNorm1d to float32 rounding. What you didn't write is the running mean and variance it also keeps for inference. Try `bn.eval()` before the last line and watch the diff jump: eval mode uses those running averages, not this batch."
mistakes:
  - { file: batchnorm/mistakes/placeholders.py, fails: per_column, says: "still returns `...`" }
  - { file: batchnorm/mistakes/return_placeholder.py, fails: per_column, says: "still returns `...`" }
  - { file: batchnorm/mistakes/rows.py, fails: per_column, says: "That's LayerNorm" }
  - { file: batchnorm/mistakes/global_stats.py, fails: per_column, says: "Each column needs its own statistics" }
  - { file: batchnorm/mistakes/not_centered.py, fails: per_column, says: "Subtract the mean before dividing" }
  - { file: batchnorm/mistakes/no_sqrt.py, fails: per_column, says: "You divided by the variance" }
  - { file: batchnorm/mistakes/unbiased.py, fails: per_column, says: "Bessel's correction" }
  - { file: batchnorm/mistakes/swapped_affine.py, fails: per_column, says: "gamma and beta are swapped" }
  - { file: batchnorm/mistakes/no_eps.py, fails: eps_guard, says: "The `eps` argument exists for exactly this" }
  - { file: batchnorm/mistakes/no_affine.py, fails: gamma_beta, says: "Apply them after normalizing" }
```

### Backprop by hand

```challenge
id: manual_backprop
file: backprop.py
minutes: 40
start: { code: "import torch\nimport torch.nn.functional as F\n\n" }
setup:
  - say: "Last, the part autograd hides. A two-layer MLP, small enough to check every number."
    type_file: manual_backprop/data.py
    pause: 2.0
  - say: "manual_backward redoes the forward pass, then walks it in reverse. Every gradient has the shape of the thing it's the gradient of, which settles most transposes. Your turn: fill in all seven."
    type_file: manual_backprop/stub.py
    pause: 1.6
tests: manual_backprop/tests.py
solution: manual_backprop/solution.py
alternatives:
  - manual_backprop/alternatives/karpathy.py
  - manual_backprop/alternatives/one_minus_tanh.py
hints:
  logic: |
    dlogits: softmax of each row, minus the right answers
      → scaled because the loss is an average
    second layer (logits came from h, W2, b2):
      dW2: the layer's input (flipped) times the gradient
           from above, shaped like W2
      db2: the gradient from above, added up over the batch
      dh:  the gradient from above sent back through W2
    through tanh, one number at a time:
      dhpreact: tanh's local slope, written with its output,
                times the gradient arriving from above
    first layer, same pattern as the second:
      dW1 and db1 from X and dhpreact, shaped like W1 and b1
  tools: |
    softmax, and the table of right answers
      F.softmax(t, dim)             softmax along dim
      F.one_hot(idx, num_classes)   0/1 table, one row per idx
      t.shape[i]                    the size of dimension i
    products and flips
      a @ b                         matrix product
      t.T                           transpose
    adding up over the batch
      t.sum(dim)                    adds along dim
    one number at a time
      plain arithmetic: * / - and ** (power)
  nudge: "Go top to bottom. Cross-entropy's gradient is the softmax minus the one-hot target, averaged over the batch. For `out = a @ W + b`: W's gradient pairs a with d(out), a's gradient pairs d(out) with W, and b's gradient sums d(out) over the batch. tanh's slope is 1 − h²."
  concept: "Shapes decide the products. W2 is [hidden, vocab], h is [n, hidden], dlogits is [n, vocab], so dW2 = h.T @ dlogits and dh = dlogits @ W2.T. A bias was added to every row, so its gradient sums over dim 0. tanh is elementwise, so its backward is `*`, not `@`. `F.one_hot(Y, vocab)` builds the target table."
  example: |
    Trace the first lines on a tiny batch: `n = 2`, 3 classes, `logits = [[0, 1, 2], [2, 0, 0]]`, `Y = [2, 0]`:

    - Softmax rows: `[0.090, 0.245, 0.665]` and `[0.787, 0.107, 0.107]`.
    - Subtract the one-hot rows `[0, 0, 1]` and `[1, 0, 0]`: `[[0.090, 0.245, -0.335], [-0.213, 0.107, 0.107]]`.
    - Divide by `n = 2`: `dlogits = [[0.045, 0.122, -0.167], [-0.107, 0.053, 0.053]]`.
    - Each row sums to 0. The right class gets a negative entry, which means raising its logit lowers the loss.
    - Summed down the batch, you get one number per class: `[-0.061, 0.176, -0.114]`. That has the shape of `b2`.

    Tanh: if one `h` entry is `0.5`, its slope is `1 - 0.25 = 0.75`. With `dh = 0.2` that entry, `dhpreact`, is `0.15`.

    Shapes for the test (`n = 5`, 3 inputs, 4 hidden, 6 classes): `dW2` is `[4, 6]`, `db2` is `[6]`, `dh` and `dhpreact` are `[5, 4]`, `dW1` is `[3, 4]`, `db1` is `[4]`.
  skeleton: |
    dlogits = (F.softmax(______, 1) - F.one_hot(_, logits.shape[1])) / _
    dW2 = _.T @ _______
    db2 = _______.sum(_)
    dh = dlogits @ __.T
    dhpreact = (1 - h**2) * __
    dW1 = X.T @ ________
    db1 = ________.sum(_)
  solution: |
    dlogits = (F.softmax(logits, 1) - F.one_hot(Y, logits.shape[1])) / n
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)
    dh = dlogits @ W2.T
    dhpreact = (1 - h**2) * dh
    dW1 = X.T @ dhpreact
    db1 = dhpreact.sum(0)
on_solve: "approximate: True everywhere, with maxdiffs of a few times 1e-8: float32 rounding, not a bug. That's every gradient autograd computed, by hand. Karpathy's backprop-ninja exercise goes further, through BatchNorm and the embedding lookup. Record your largest maxdiff."
mistakes:
  - { file: manual_backprop/mistakes/placeholders.py, fails: dlogits, says: "`dlogits` is still `...`" }
  - { file: manual_backprop/mistakes/dW2_no_transpose.py, fails: runs, says: "Which product of the two, with one of them transposed, is [4, 6]?" }
  - { file: manual_backprop/mistakes/dh_no_transpose.py, fails: runs, says: "Which product of the two, with one of them transposed, is [5, 4]?" }
  - { file: manual_backprop/mistakes/dlogits_no_n.py, fails: dlogits, says: "5 times too big" }
  - { file: manual_backprop/mistakes/dlogits_softmax_only.py, fails: dlogits, says: "gets 1 subtracted" }
  - { file: manual_backprop/mistakes/dlogits_flipped.py, fails: dlogits, says: "The sign is flipped" }
  - { file: manual_backprop/mistakes/db2_wrong_dim.py, fails: layer2, says: "you summed over the classes" }
  - { file: manual_backprop/mistakes/db2_mean.py, fails: layer2, says: "not their average" }
  - { file: manual_backprop/mistakes/dhpreact_input.py, fails: through_tanh, says: "You used hpreact" }
  - { file: manual_backprop/mistakes/dhpreact_no_chain.py, fails: through_tanh, says: "local derivative on its own" }
  - { file: manual_backprop/mistakes/dW1_transposed.py, fails: layer1, says: "the transpose of `W1`" }
  - { file: manual_backprop/mistakes/db1_placeholder.py, fails: layer1, says: "`db1` is still `...`" }
```

### Train the MLP with the fixes

```replay
id: mlp_init_200k
intro: Five 200k-step runs of the makemore MLP take a few minutes each on a GPU. Here's an illustrative log so you know what to expect from each fix.
log: replays/mlp-init-200k.jsonl
speed: 30
recorded:
  hardware: RTX 3090
  illustrative: true
  note: Sample output written for the course, close to the numbers in Karpathy's makemore part 3. Not a recording. Your exact values will differ with the seed.
chart:
  title: Validation loss after 200k steps
  flag_above: 2.15
  x_labels: [original, batchnorm]
  caption: "The original init wastes its first steps undoing confident, saturated outputs, and ends worse. Each fix gets to about 2.10. BatchNorm matters less in a network this shallow, and a lot more in deep ones."
summary:
  - { label: Step-0 loss, value: 27.9 → 3.3 }
  - { label: Tanh saturated at init, value: 79% → 4% }
  - { label: Val loss, value: 2.168 → 2.103 }
run_yourself: |
  # Karpathy's makemore part 3 MLP on all of names.txt: block_size 3, 10-dim embedding, 200 tanh units,
  # batch 32, lr 0.1 then 0.01 after 100k steps. Run it once per configuration:
  #   original    W1, W2 from plain torch.randn
  #   fix-init    W2 *= 0.01, b2 *= 0, W1 from your linear_weights(30, 200, g, gain=5/3)
  #   batchnorm   drop b1, put your batchnorm (with running stats for eval) between W1 and tanh
  python mlp_init.py --steps 200000 --config fix-init
```

### Record your metrics

```metrics
id: record
intro: The validation loss from your own best 200k-step run, and the largest maxdiff from step 3.
```

## Hints

- Check the step-0 loss before training anything. On 27 symbols it should be close to ln 27 ≈ 3.30. If it's far above, the output layer is too confident.
- Plot a histogram of the hidden tanh outputs at init: `plt.hist(h.view(-1).tolist(), 50)`. Big spikes at ±1 mean saturated units.
- Running stats for inference: `running_mean = 0.999 * running_mean + 0.001 * mean` each step, under `torch.no_grad()`. Use them, not the batch's statistics, when you sample names.
- When a hand-written gradient is wrong, compare shapes first. A gradient always has the shape of the tensor it belongs to.

## Done when

- [ ] Weights come out with std gain / √fan_in {#kaiming auto=challenge:kaiming_init}
- [ ] Your batchnorm matches nn.BatchNorm1d {#batchnorm auto=challenge:batchnorm}
- [ ] All seven hand-written gradients match autograd {#manual_backprop auto=challenge:manual_backprop}
- [ ] Validation loss from your own 200k-step run and the manual-backprop error recorded {#metrics_recorded auto=metrics}
