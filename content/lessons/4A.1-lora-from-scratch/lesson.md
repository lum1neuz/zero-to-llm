---
id: "4A.1"
title: LoRA from scratch
summary: Freeze a layer, train two thin matrices beside it, and fine-tune your 30M model with about 1% of its parameters.
phase: "4A"
hours: 5
type: build
status: ready
protected: false
cut: null
prerequisites:
  - "3.6"
papers:
  - title: LoRA
    authors: Hu et al.
    year: 2021
    url: https://arxiv.org/abs/2106.09685
quality_metric:
  id: val_loss
  label: Validation loss after LoRA fine-tune
  unit: nats
  note: TinyStories-Instruct val split, r = 8 on all 24 Linear layers of your 30M model. The base model starts around 2.3.
  inputs:
    - { id: val_loss, label: LoRA val loss (nats), hint: The last val line of your run }
  compute: value
  reference: 1.86
  pass: { max: 2.0 }
systems_metric:
  id: trainable_pct
  label: Trainable parameters
  unit: "%"
  note: r × (in + out) summed over the 24 layers is 393,216, out of 30.6M including the adapters.
  inputs:
    - { id: trainable_pct, label: trainable% (from the run's first lines), hint: "PEFT prints it as trainable%" }
  compute: value
  reference: 1.29
  pass: { max: 2.0 }
quiz:
  - id: both_zero
    q: LoRA starts A random and B at zero. What goes wrong if both start at zero?
    answer: 0
    options:
      - text: Nothing ever trains, because each one's gradient runs through the other
        why: B's gradient is built from x @ Aᵀ and A's from B. With both at zero, both gradients are exactly zero on every step, so neither ever moves.
      - text: The model starts out different from the base model
        why: B @ A is zero either way, so the start is identical. The problem shows up in the gradients, not the output.
      - text: Nothing. Adam rescales tiny gradients, so they still get moving
        why: Adam divides by a running gradient size, but a gradient of exactly zero stays zero after rescaling (eps keeps it from being 0 / 0). Zero in, zero out.
  - id: alpha_over_r
    q: You keep alpha = 16 and raise the rank r from 8 to 32. What does the alpha / r scale do?
    answer: 1
    options:
      - text: Nothing changes. alpha is the scale
        why: The scale is alpha / r, so it moves with r. With alpha alone, a higher rank would also mean a bigger update.
      - text: It drops from 2 to 0.5, so a higher rank doesn't mean a bigger update
        why: A rank-32 B @ A sums 32 directions instead of 8. Dividing by r keeps the update's size about the same, so the learning rate you tuned at r = 8 still works.
      - text: It makes the adapter 4× stronger
        why: The rank grows 4×, but the scale shrinks 4×. You get more directions to learn in, not a bigger push.
  - id: merge_cost
    q: After training you merge, W′ = W + (alpha / r) · B @ A, and throw A and B away. What does the merged model cost at inference?
    answer: 2
    options:
      - text: One extra small matmul per layer
        why: That's the unmerged adapter, x @ Aᵀ @ Bᵀ beside the base. Merging folds it into W, so the extra matmul is gone.
      - text: A few percent more memory for the adapter weights
        why: W′ has exactly W's shape. Once merged, A and B can be dropped.
      - text: The same as the base model, because it has the same shapes and matmuls
        why: Merging changes W's values, not its shape. Same memory, same speed. (With a 4-bit base in 4A.2 you can't merge without re-quantizing, which is one reason adapters are often kept separate.)
---

## Goal

Fine-tune without touching the pretrained weights. Freeze a layer's weight `W`, put two thin trainable matrices beside it, `A` (`[r, in]`) and `B` (`[out, r]`), and let the layer compute `W x + (alpha / r) · B A x`. Build that layer by hand, check that it starts exactly at the base model, train only `A` and `B` with mini-torch, then wrap every Linear in your 30M model and fine-tune it with about 1% of its parameters.

## Learn

```card
type: concept
id: low_rank_update
eyebrow: The idea
title: Fine-tune a 4096 × 4096 layer by training two thin matrices.
visual:
  kind: bars
  max: 16.8
  bars:
    - { label: full, value: 16.78, text: 16.8M }
    - { label: r = 64, value: 0.524, text: 524k }
    - { label: r = 16, value: 0.131, text: 131k }
body: |
  A full fine-tune updates every number in `W`. LoRA freezes `W` and learns the change as a product, **ΔW = B @ A**, where A is `[r, in]` and B is `[out, r]`. With a small rank r, that's r × (in + out) numbers instead of in × out. The bars count trainable parameters for one 4096 × 4096 projection.
```

```card
type: fill_blank
id: lora_forward
eyebrow: Fill the blanks
title: Complete the LoRA forward pass.
code: |
  def forward(self, x):
      base = self.base(x)
      update = x @ self.[[0]].T @ self.[[1]].T
      return base + update * [[2]]
tokens: [A, B, W, alpha / r, r / alpha, alpha]
answer: [A, B, alpha / r]
tips:
  W: "W is frozen and already inside `self.base(x)`. The update only goes through the two thin matrices."
  r / alpha: Flipped. LoRA multiplies the update by alpha / r.
  alpha: Without dividing by r, doubling the rank doubles the update's size, and the learning rate you tuned stops fitting.
why_ok: "`x` (`[batch, in]`) meets A first and shrinks to r numbers per row, then B expands them to `out`. The scale alpha / r keeps the update's size steady across ranks."
why_wrong: "Follow the shapes: `x` is `[batch, in]`, A is `[r, in]`, B is `[out, r]`. Which one can `x` meet first?"
hint: A squeezes the input down to r numbers, B expands them back out.
```

```card
type: predict
id: zero_start
eyebrow: Predict the output
title: B starts at zero. What prints?
code: |
  base = nn.Linear(4, 4)
  lora = LoRALinear(base, r=2, alpha=4)   # A random, B zeros
  x = torch.randn(3, 4)
  print(torch.equal(lora(x), base(x)))
options:
  - text: "True"
    correct: true
    why: B is all zeros, so x @ Aᵀ @ Bᵀ is all zeros and adding it changes nothing, bit for bit. Fine-tuning starts from exactly the pretrained model.
  - text: "False, because the random A nudges the output a little"
    why: A's output gets multiplied by B, and B is all zeros. However random A is, B @ A is the zero matrix.
  - text: "An error: B @ A doesn't match W's shape"
    why: B is [4, 2] and A is [2, 4], so B @ A is [4, 4], the same shape as W. That's what lets the two add.
hint: What is any matrix multiplied by a matrix of zeros?
```

```card
type: calc
id: trainable_count
eyebrow: Do the math
title: How many trainable parameters does r = 16 add to one q_proj of Llama 3.1 8B?
given:
  - { label: q_proj shape, value: 4096 in → 4096 out }
  - { label: Rank r, value: "16" }
  - { label: A is, value: "[r, in]" }
  - { label: B is, value: "[out, r]" }
unit: params
answer: { min: 131072, max: 131072 }
placeholder: "0"
why_ok: 16 × 4096 + 4096 × 16 = 131,072, which is 0.78% of the 16.8M weights it adapts. On q, k, v, o, gate, up and down in all 32 layers it comes to 41.9M, 0.52% of the model.
why_wrong: Count both matrices. A has r × in numbers, B has out × r. Neither is anywhere near in × out.
hint: Add A's size to B's size. r × (in + out).
```

```card
type: reorder
id: lora_run
eyebrow: Put it in order
title: Arrange a LoRA fine-tune, start to finish.
lines:
  - lora = LoRALinear(base, r=4, alpha=8)
  - opt = torch.optim.Adam(trainable(lora), lr=0.05)
  - "for step in range(200): train_step(lora, opt, X, Y)"
  - W_merged = base.weight + lora.scale * lora.B @ lora.A
  - print((X @ W_merged.T + base.bias - lora(X)).abs().max())
shuffle: [3, 0, 4, 2, 1]
why_ok: "Wrap the layer, give the optimizer only A and B, train, then merge. The merge folds the trained, scaled B @ A into W, and the last line checks that the merged layer gives the same output as the adapter, up to float rounding."
why_wrong: "Something runs before what it needs exists. The optimizer needs the wrapped model, the merge needs trained A and B, and the check needs the merged weight."
hint: "Follow the names, and merge only once A and B have learned something."
```

```card
type: quiz
ref: both_zero
```

```card
type: quiz
ref: alpha_over_r
```

```card
type: quiz
ref: merge_cost
```

## Build steps

### Write a LoRA layer

```challenge
id: lora_layer
file: lora.py
minutes: 35
start: { code: "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n" }
setup:
  - say: Here's an ordinary Linear layer, 6 inputs to 10 outputs. Pretend it's one projection inside a pretrained model.
    type: "torch.manual_seed(0)\nbase = nn.Linear(6, 10)\nx = torch.randn(3, 6)\n\n"
    pause: 2.4
  - say: "LoRA never changes its weights. Two thin matrices sit beside it: A squeezes each input down to r numbers, B expands them back out, and only those two train."
    pause: 4.0
  - say: "Here's the stub. Your turn: freeze the base, create A and B, pick the scale, and write forward. The two prints at the bottom check your work."
    type_file: lora_layer/stub.py
    pause: 1.2
tests: lora_layer/tests.py
solution: lora_layer/solution.py
alternatives:
  - lora_layer/alternatives/merged.py
  - lora_layer/alternatives/loop_freeze.py
  - lora_layer/alternatives/f_linear.py
  - lora_layer/alternatives/kaiming.py
  - lora_layer/alternatives/weight_shape.py
hints:
  start:
    try: |
      print(base.in_features, base.out_features)
      print(base(x).shape)
      print(base.weight.requires_grad)
    see: |
      6 10
      torch.Size([3, 10])
      True
    next: "That's the base layer's size (6 in, 10 out), the shape its output has for 3 inputs, and that its weights can still train. The update from `A` and `B` must also come out [3, 10], so `A` is `r` by 6 and `B` is 10 by `r`. The first line to write freezes the base, so that last print becomes False."
  logic: |
    setup:
      freeze everything the base layer owns
      A = small random numbers, shape r × in
      B = zeros, shape out × r   (so the update starts at nothing)
      scale = alpha ÷ r
    forward on x:
      base output = the frozen layer applied to x
      x through A (r numbers) → through B (out numbers)
        → times scale = the update
      return base output + update
  tools: |
    freeze the base layer
      module.requires_grad_(flag)   turns grads on or off for
                                    every parameter it owns
    make A and B trainable
      nn.Parameter(t)    registers t as a trainable parameter
      torch.randn(a, b)  random normal numbers, shape [a, b]
      torch.zeros(a, b)  zeros, shape [a, b]
      layer.in_features, layer.out_features   a Linear's sizes
    scale
      plain arithmetic: /
    forward
      a @ b              matrix product
      t.T                rows and columns swapped
      plain arithmetic: * +
  nudge: "Four pieces: switch off gradients for everything the base owns, make A small and random, make B zero, and add the scaled update to the base's output. Which shapes let `x @ A.T @ B.T` work?"
  concept: "`module.requires_grad_(False)` freezes every parameter a module owns, weight and bias. `nn.Parameter(t)` registers a tensor so `parameters()` and the optimizer can see it. A is `[r, in]` and B is `[out, r]`, stored like `nn.Linear` weights, so `x @ A.T` is `[batch, r]` and `@ B.T` makes it `[batch, out]`. B starts at zero so that B @ A is zero and step 0 is exactly the base model. The scale is alpha / r."
  example: |
    Trace the lesson's layer: `base = nn.Linear(6, 10)`, `r = 4`, `alpha = 8`, and `x` of shape [3, 6].

    - `A` is [4, 6] (24 numbers) and `B` is [10, 4] (40 numbers). The scale is 8 / 4 = `2.0`.
    - Shapes in `forward`: `x @ A.T` is [3, 4], then `@ B.T` is [3, 10], the same as `base(x)`, so the two add.
    - At step 0, `B` is all zeros, so the update is all zeros and the output is exactly `base(x)`. That's the first print: `True`.
    - Only `A` and `B` still require grad, so the second print lists `A` and `B` and nothing starting with `base.`.

    A tiny version with numbers, `r = 1`, `x = [[1, 2, 3]]`, `A = [[0.01, -0.02, 0.03]]`: `x @ A.T = [[0.06]]`. If `B` later becomes `[[0.5], [-1.0]]` and the scale is 2, the update is `0.06 × [0.5, -1.0] × 2 = [[0.06, -0.12]]`.
  skeleton: |
    self.base.______________(False)
    self.A = nn.Parameter(torch.randn(___, base.in_features) * 0.01)
    self.B = nn.Parameter(torch._____(base.out_features, ___))
    self.scale = _____ / ___
    return self.base(x) + (x @ self.A.T @ self.B.T) * __________
  solution: |
    self.base.requires_grad_(False)
    self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
    self.B = nn.Parameter(torch.zeros(base.out_features, r))
    self.scale = alpha / r
    return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
on_solve: Solved. Your layer is exactly the base model until B moves, and only A and B can move. Step 2 is unlocked. Train them.
mistakes:
  - { file: lora_layer/mistakes/placeholders.py, fails: builds, says: "`self.A`, `self.B` and `self.scale` are still `...`" }
  - { file: lora_layer/mistakes/plain_tensor.py, fails: builds, says: "is a plain tensor, so `model.parameters()` never lists it" }
  - { file: lora_layer/mistakes/a_transposed.py, fails: shapes, says: "that's [in, r]" }
  - { file: lora_layer/mistakes/not_frozen.py, fails: base_frozen, says: "`base.weight` and `base.bias` still require grad" }
  - { file: lora_layer/mistakes/weight_only.py, fails: base_frozen, says: "`base.bias` still requires grad" }
  - { file: lora_layer/mistakes/b_random.py, fails: starts_as_base, says: "B starts random" }
  - { file: lora_layer/mistakes/no_base.py, fails: starts_as_base, says: "returns only the update" }
  - { file: lora_layer/mistakes/weight_no_bias.py, fails: starts_as_base, says: "without the bias" }
  - { file: lora_layer/mistakes/matmul_order.py, fails: starts_as_base, says: "Which of the two can `x` meet first" }
  - { file: lora_layer/mistakes/forward_placeholder.py, fails: starts_as_base, says: "`forward` still returns `...`" }
  - { file: lora_layer/mistakes/no_scale.py, fails: scaled_update, says: "Your update isn't scaled" }
  - { file: lora_layer/mistakes/r_over_alpha.py, fails: scaled_update, says: "Flip it" }
  - { file: lora_layer/mistakes/alpha_only.py, fails: scaled_update, says: "not alpha / r" }
  - { file: lora_layer/mistakes/hardcoded_scale.py, fails: scaled_update, says: "Your scale is 2 no matter what" }
  - { file: lora_layer/mistakes/both_zero.py, fails: grads_flow, says: "A and B both start at zero" }
```

### Train only A and B

```challenge
id: train_lora
file: lora.py
minutes: 25
start: { from: lora_layer }
setup:
  - say: Your layer starts exactly at the base model. Now give it something to learn, the same layer plus a rank-2 change, the kind of small shift a fine-tune makes.
    type: "\ntorch.manual_seed(1)\nX = torch.randn(64, 6)\ndelta = torch.randn(10, 2) @ torch.randn(2, 6) * 0.5\nY = base(X) + X @ delta.T\n\n"
    pause: 2.6
  - say: "Two functions to write: which parameters the optimizer gets, and one training step. The loop at the bottom runs 200 steps and checks the base weights never moved."
    type_file: train_lora/stub.py
    pause: 1.2
tests: train_lora/tests.py
solution: train_lora/solution.py
alternatives:
  - train_lora/alternatives/explicit_mse.py
  - train_lora/alternatives/zero_at_end.py
  - train_lora/alternatives/filter.py
  - train_lora/alternatives/model_zero_grad.py
hints:
  start:
    try: |
      for name, p in model.named_parameters():
          print(name, p.requires_grad)
    see: |
      A True
      B True
      base.weight False
      base.bias False
    next: "That's the model's four parameters: the base layer's weight and bias, frozen (False), and `A` and `B`, still training (True). `trainable` returns only the True ones, as a list. For `train_step`, compute the mean squared error of `model(X)` against `Y` and print it first, before anything touches the optimizer."
  logic: |
    trainable:
      for each parameter of the model:
        still requires grad?
          yes → keep it
          no  → skip it
      return the kept ones as a list
    train_step:
      clear the old gradients
        → run the model on X → loss = mean squared error vs Y
        → backpropagate → optimizer takes a step
        → return the loss as a number
  tools: |
    keep only the trainable parameters
      model.parameters()   every parameter of the model
      p.requires_grad      True while p still trains
      [x for x in xs if c] a list of the items that pass c
    clear old gradients
      optim.zero_grad()    sets every gradient to zero
    the loss
      F.mse_loss(pred, y)  mean squared error
    learn
      y.backward()         fills the gradients
      optim.step()         moves the parameters one step
  nudge: "`trainable` keeps a parameter only if it still requires grad. `train_step` is the loop you've written since micrograd: clear, forward and loss, backward, step."
  concept: "`model.parameters()` yields every parameter, frozen ones included, so filter on `p.requires_grad`, and make it a list because the script reads it twice. `backward()` adds into `.grad`, so `opt.zero_grad()` has to run before it. `F.mse_loss(pred, target)` is the mean of the squared errors."
  example: |
    Trace the lesson's model, `base = nn.Linear(6, 10)` with `r = 4`.

    - `model.parameters()` yields 4 tensors: `base.weight` [10, 6], `base.bias` [10], `A` [4, 6], `B` [10, 4]. The first two have `requires_grad = False`, so `trainable` returns a list of 2: `A` and `B`.
    - Counts: 24 + 40 = 64 trainable of 134 in total (60 + 10 for the base), which prints as `47.8%`.
    - Mean squared error on a tiny case: predictions `[1, 3]` against targets `[0, 1]` give (1² + 2²) / 2 = `2.5`. A sum would give 5.0.
    - The loss `train_step` returns is measured before the optimizer moves anything, so step 0 prints the biggest loss, and the later prints fall toward 0. The base weights stay equal to `w0`, because the optimizer never saw them.
  skeleton: |
    return [p for p in model.__________() if p.____________]
    opt.__________()
    loss = F.________(model(X), Y)
    loss.________()
    opt.____()
  solution: |
    return [p for p in model.parameters() if p.requires_grad]
    opt.zero_grad()
    loss = F.mse_loss(model(X), Y)
    loss.backward()
    opt.step()
on_solve: "The loss falls to about zero in 100 steps. The target is a rank-2 change and your adapter has rank 4, so it can hold it exactly. Now try `r=1` in the `model = LoRALinear(...)` line: one direction can't hold a rank-2 change, and the loss stalls. And look at the 47.8%: on a 6 × 10 layer LoRA saves almost nothing. The savings come from r being far smaller than the layer. At 4096 × 4096 with r = 16 it's 0.78%."
mistakes:
  - { file: train_lora/mistakes/placeholders.py, fails: trainable_list, says: "`trainable` still returns `...`" }
  - { file: train_lora/mistakes/all_params.py, fails: trainable_list, says: "which are frozen" }
  - { file: train_lora/mistakes/generator.py, fails: trainable_list, says: "returns a generator" }
  - { file: train_lora/mistakes/only_b.py, fails: trainable_list, says: "leaves out `A`" }
  - { file: train_lora/mistakes/named.py, fails: trainable_list, says: "(name, parameter) pairs" }
  - { file: train_lora/mistakes/loss_placeholder.py, fails: step_trains, says: "`loss` is still `...`" }
  - { file: train_lora/mistakes/sum_loss.py, fails: step_trains, says: "*sum* of squared errors" }
  - { file: train_lora/mistakes/mae.py, fails: step_trains, says: "mean *absolute* error" }
  - { file: train_lora/mistakes/no_step.py, fails: step_trains, says: "`opt.step()` never runs" }
  - { file: train_lora/mistakes/no_backward.py, fails: step_trains, says: "Which call fills `.grad`" }
  - { file: train_lora/mistakes/zero_grad_late.py, fails: step_trains, says: "after `backward()` and before `opt.step()`" }
  - { file: train_lora/mistakes/no_zero_grad.py, fails: step_trains, says: "Gradients pile up" }
```

### LoRA on your 30M model

```replay
id: lora_30m
intro: "On your GPU, every nn.Linear in the 30M model gets wrapped in your LoRALinear (r = 8, alpha = 16) and fine-tuned on TinyStories-Instruct, stories with a prompt of features, words and a summary. It takes about 15 minutes. Watch a recording first so you know what healthy looks like."
log: replays/lora-30m.jsonl
speed: 40
recorded:
  hardware: RTX 3060 12 GB
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: Val loss at each eval
  flag_above: 2.0
  x_labels: [step 0, step 3000]
  caption: Red bars sit above 2.0 nats. Most of the gain comes in the first 1000 steps. A full fine-tune of the same model ends at 1.79. LoRA gets to 1.86 while training 1.3% of the parameters.
summary:
  - { label: Trainable, value: "393k of 30.6M (1.29%)" }
  - { label: Val loss, value: 2.31 → 1.86 }
  - { label: Full fine-tune, value: "1.79" }
run_yourself: |
  # on your machine or Colab, from the course repo
  python lora_finetune.py \
    --ckpt runs/tinystories-30m/best.pt \
    --data tinystories-instruct --rank 8 --alpha 16 --steps 3000
```

### Record your metrics

```metrics
id: record
intro: Your LoRA run's final val loss and its trainable share. They're compared with the reference run and the pass limit.
```

## Hints

- Freeze before you train, and give the optimizer only the trainable parameters. Adam allocates no state for weights that never get a gradient, but passing just the LoRA parameters makes it obvious what you're training and catches a forgotten `requires_grad=False`.
- PEFT names these `lora_A` and `lora_B` and stores them exactly as here, `[r, in]` and `[out, r]`. Its default init for A is Kaiming-uniform rather than small Gaussian noise. Either works, as long as B starts at zero.
- Common settings: r = 8 to 64, alpha = 2r (a scale of 2), LoRA on every Linear rather than only q and v. The original paper adapted only q and v; later work (QLoRA among it) found every Linear works better for the same budget.
- If the val loss doesn't move at all, check that B gets a gradient. A forward that detaches or skips the update trains nothing and raises no error.

## Done when

- [ ] Your LoRA layer starts exactly at the base model and only A and B can train {#lora_layer auto=challenge:lora_layer}
- [ ] 200 steps fit the rank-2 target while the base weights stay untouched {#train_lora auto=challenge:train_lora}
- [ ] Your 30M model is fine-tuned with r = 8 adapters and both metrics are recorded {#metrics_recorded auto=metrics}
