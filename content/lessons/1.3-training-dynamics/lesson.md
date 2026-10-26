---
id: "1.3"
title: Training dynamics
summary: Write a warmup-plus-cosine learning-rate schedule and global gradient clipping by hand, then see what they do to a real training run.
phase: "1"
hours: 4
type: run
status: ready
protected: false
cut: null
prerequisites:
  - "1.2"
papers: []
quality_metric:
  id: val_loss
  label: Validation loss with LR schedule and clipping
  unit: "%"
  note: The change in Shakespeare val loss from 1.2's constant learning rate to warmup, cosine decay and clipping. Negative is better.
  inputs:
    - { id: val_constant, label: Val loss with a constant LR (from 1.2), hint: Your 1.2 run }
    - { id: val_schedule, label: Val loss with warmup + cosine + clip, hint: The last eval line of this run }
  compute: pct_change
  reference: -3.3
  pass: { max: 0 }
systems_metric:
  id: tokens_per_s
  label: Training throughput
  unit: tok/s
  note: 64 × 256 tokens per step on an RTX 3090. A laptop CPU manages a few thousand; record it anyway.
  inputs:
    - { id: tokens_per_s, label: Tokens per second, hint: Printed with every eval }
  compute: value
  reference: 310000
  pass: { min: 20000 }
quiz:
  - id: why_warmup
    q: Why start with a tiny learning rate for the first hundred or so steps?
    answer: 2
    options:
      - text: To save compute while the loss is still high
        why: A step costs the same compute whatever the learning rate. Warmup is about where the steps go, not what they cost.
      - text: Because the loss is already small at the start
        why: The loss is at its largest at the start, about ln(65) ≈ 4.2 for 65 characters.
      - text: Early gradients and Adam's running averages are unreliable, so full-size steps can throw the weights somewhere they never recover from
        why: At step 0, Adam's estimate of each gradient's scale comes from a single batch. Small steps while those estimates settle keep one noisy start from wrecking the run.
      - text: So the causal mask has time to take effect
        why: The mask is fixed from step 0. It doesn't learn anything.
  - id: loss_spike
    q: "At step 2,750 the grad norm jumps from 0.3 to 3.4 for one batch. With clip_grad_norm_(params, 1.0), what happens to that step?"
    answer: 1
    options:
      - text: The batch is skipped
        why: Clipping never skips a batch. It still takes the step, just a shorter one.
      - text: The gradient is scaled down 3.4× but keeps its direction, so one bad batch can't throw the weights far
        why: Every gradient is multiplied by 1.0 / 3.4 and points the same way. AdamW then rescales per parameter, so the step isn't exactly 3.4× shorter, but the spike can no longer dominate the update. On normal batches, with norms under 1, nothing changes.
      - text: The largest gradient entries are set to 1.0 and the rest are kept
        why: That's clipping by value, which bends the update's direction. clip_grad_norm_ scales everything by one shared factor.
      - text: The learning rate drops for the rest of training
        why: Clipping acts on one step's gradients. The schedule is untouched.
---

## Goal

Make training **stable** and squeeze out a better loss with two small pieces of code. Write a **learning-rate schedule** that warms up in a straight line and then glides down half a cosine to a floor, and **global gradient clipping**, which scales gradients down when their combined length is too long, without changing their direction. Then watch both at work in a recording of 1.2's Shakespeare run.

## Learn

```card
type: concept
id: schedule_shape
eyebrow: The schedule
title: Warm up, then glide down.
body: |
  A constant learning rate is a compromise: too big for the start, when the model is fragile, and too big for the end, when it needs small steps to settle. **Warmup** climbs from nearly 0 to the peak over the first ~100 steps. **Cosine decay** then glides down, slowly at first, steeply in the middle, slowly again near a floor of a tenth of the peak. The chart shows 1.2's run with this schedule.

  ```chart
  kind: line
  title: Learning rate in 1.2's run
  x: { label: step, values: [0, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000] }
  y: { label: learning rate, min: 0, max: 0.001 }
  series:
    - { name: learning rate, values: [0.00001, 0.00051, 0.001, 0.000998, 0.000985, 0.000961, 0.000927, 0.000883, 0.000831, 0.000771, 0.000705, 0.000636, 0.000564, 0.000492, 0.000422, 0.000355, 0.000293, 0.000237, 0.000189, 0.000151, 0.000123, 0.000106, 0.0001] }
  highlight: { x: 100, label: "peak, 1e−3" }
  caption: A steep climb over 100 steps, then a slow, fast, slow glide down to the 1e−4 floor.
  ```
```

```card
type: predict
id: predict_midpoint
eyebrow: Predict the output
title: max_lr 6e-4, min_lr 6e-5, warmup 10, max_steps 50. What's get_lr(30)?
code: |
  print(f"{get_lr(30):.2e}")
options:
  - text: "3.30e-04"
    correct: true
    why: Step 30 is exactly halfway through the decay (20 of 40 steps). cos(π/2) = 0, so coeff = 0.5 and the lr is halfway between the peak and the floor, (6e-4 + 6e-5) / 2.
  - text: "3.00e-04"
    why: That's halfway to 0. The cosine glides down to min_lr, not to 0, so its midpoint is halfway between 6e-4 and 6e-5.
  - text: "6.00e-04"
    why: That's the peak, reached at the end of warmup. By step 30 the decay is halfway done.
  - text: "6.00e-05"
    why: That's the floor, which the decay only reaches at max_steps, step 50.
hint: How far through the decay is step 30, if the decay runs from step 10 to step 50? What's cos(π × that)?
```

```card
type: fill_blank
id: cosine_coeff
eyebrow: Fill the blanks
title: Squeeze half a cosine into 1 → 0.
code: |
  progress = (step - warmup_steps) / (max_steps - warmup_steps)
  coeff = 0.5 * ([[0]] + math.cos(math.[[1]] * progress))
  lr = min_lr + coeff * (max_lr - min_lr)
tokens: ["1.0", "0.0", pi, e, "2.0"]
answer: ["1.0", pi]
tips:
  "0.0": "Then coeff is 0.5 × cos, which runs from 0.5 down to −0.5, and the lr ends below the floor."
  e: math.e is 2.718…, which isn't a half turn. Half a turn of the cosine is π radians.
  "2.0": Then coeff starts at 1.5, and the lr overshoots max_lr by half.
why_ok: "cos(π × progress) runs from 1 to −1 as progress goes 0 → 1. Adding 1 gives 2 → 0, and halving gives 1 → 0, so the lr glides from max_lr to min_lr."
why_wrong: "Work out coeff at progress = 0 and at progress = 1. It should be exactly 1 and exactly 0."
hint: At progress = 0 you want coeff = 1, and at progress = 1 you want coeff = 0. cos 0 = 1, and which angle has cos = −1?
```

```card
type: calc
id: global_norm
eyebrow: Do the math
title: Three parameters have gradient norms 2, 3 and 6. What's the global norm?
given:
  - { label: Gradient norms, value: "2, 3, 6" }
  - { label: Global norm, value: all gradients laid end to end as one vector }
unit: ""
answer: { min: 6.99, max: 7.01 }
placeholder: "0"
why_ok: √(2² + 3² + 6²) = √(4 + 9 + 36) = √49 = 7. Squares add, lengths don't. With max_norm 1.0, every gradient would be scaled by 1/7.
why_wrong: Lengths don't add (that would be 11). Square each norm, add the squares, and take one square root.
hint: The global norm is a square root of a sum of squares, and each tensor's squared norm is already the sum of its squares.
```

```card
type: reorder
id: step_order
eyebrow: Put it in order
title: Where does clipping go in a training step?
lines:
  - logits, loss = model(x, y)
  - loss.backward()
  - norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  - optimizer.step()
shuffle: [2, 3, 0, 1]
why_ok: "Clipping needs the gradients, so it comes after backward(), and it has to change them before optimizer.step() uses them."
why_wrong: "Clipping acts on gradients: they don't exist before backward(), and after step() it's too late."
hint: What does clipping read, and which line consumes what it changes?
```

```card
type: quiz
ref: why_warmup
```

```card
type: quiz
ref: loss_spike
```

## Build steps

### Write the learning-rate schedule

```challenge
id: get_lr
file: train.py
minutes: 25
start: { code: "import math\n\n" }
setup:
  - say: Here's the schedule for a short run. A peak learning rate, a floor at a tenth of it, 10 warmup steps and 50 steps in all.
    type: "max_lr = 6e-4\nmin_lr = max_lr * 0.1\nwarmup_steps = 10\nmax_steps = 50\n\n"
    pause: 2.6
  - say: "Warm up in a straight line, glide down half a cosine, then hold at the floor."
    pause: 2.4
  - say: "Here's the stub. Your turn: fill in the two returns, progress and coeff."
    type_file: get_lr/stub.py
    pause: 1.2
tests: get_lr/tests.py
solution: get_lr/solution.py
alternatives:
  - get_lr/alternatives/from_zero.py
  - get_lr/alternatives/halved.py
  - get_lr/alternatives/min_lr_first.py
hints:
  start:
    try: |
      print(f"max {max_lr:.1e}  min {min_lr:.1e}")
      print("step 30 is", 30 - warmup_steps, "steps into a decay of", max_steps - warmup_steps)
    see: |
      max 6.0e-04  min 6.0e-05
      step 30 is 20 steps into a decay of 40
    next: "That's step 30 sitting 20 steps into a 40-step decay, halfway down. `progress` is that fraction, running from 0 to 1. Write it for step 30 first and check you get 0.5, then `coeff` is half a cosine of that progress, falling from 1 to 0."
  logic: |
    step < warmup_steps?
      yes → climb in a straight line
              the last warmup step reaches max_lr
            → return that
      no  → step > max_steps?
              yes → hold at the floor, min_lr
              no  → progress = how far through the decay
                      (0 at warmup_steps, 1 at max_steps)
                    → coeff = half a cosine, from 1 down to 0
                    → blend between min_lr and max_lr by coeff
                    → return the blend
  tools: |
    warmup: a straight climb
      plain arithmetic: * / +
    decay progress, 0 → 1
      plain arithmetic: - /
    half a cosine, 1 down to 0
      math.cos(x)         cosine of x (in radians)
      math.pi             the constant π
      plain arithmetic: * +
  nudge: Three phases. During warmup, what fraction of max_lr should the last of 10 warmup steps get? During the decay, what should progress be at its first and its last step?
  concept: "Warmup climbs in equal steps and reaches max_lr on its last step, so step 9 of 10 gets the full max_lr. In the decay, progress runs from 0 at the end of warmup to 1 at max_steps. cos(π × progress) runs from 1 to −1; 0.5 × (1 + cos) squeezes that into 1 → 0."
  example: |
    Trace it with the script's numbers: `max_lr = 6e-4`, `min_lr = 6e-5`, 10 warmup steps, 50 steps in all:

    - Step 4 is in warmup: 5 of 10 steps, so half of `max_lr`, `3.00e-04`. Step 9 gets the full `6.00e-04`.
    - Step 10: `progress = 0`, `coeff = 1`, so the lr is `max_lr`, `6.00e-04`.
    - Step 30: `progress = 20 / 40 = 0.5`, cosine of π/2 is 0, so `coeff = 0.5`. The lr is the midpoint of 6e-5 and 6e-4: `3.30e-04`.
    - Step 50: `progress = 1`, `coeff = 0`, so the lr is `min_lr`, `6.00e-05`. Step 60 is past `max_steps`: also `6.00e-05`.
  skeleton: |
    return max_lr * (____ + 1) / ____________
    return ______
    progress = (step - ____________) / (_________ - ____________)
    coeff = 0.5 * (1.0 + math.cos(math.__ * ________))
  solution: |
    return max_lr * (step + 1) / warmup_steps
    return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
on_solve: "Read the printed lrs: 6.0e−5 at step 0, the peak by step 9, the midpoint 3.3e−4 at step 30, the floor from step 50 on. Step 2 is unlocked: the other half of a stable run."
mistakes:
  - { file: get_lr/mistakes/placeholders.py, fails: warmup, says: "a `return ...` placeholder is still there" }
  - { file: get_lr/mistakes/no_warmup.py, fails: warmup, says: "so there's no warmup" }
  - { file: get_lr/mistakes/no_max_lr.py, fails: warmup, says: "Scale it by max_lr" }
  - { file: get_lr/mistakes/zero_after.py, fails: decay_ends, says: "training stops learning" }
  - { file: get_lr/mistakes/progress_from_zero.py, fails: decay_ends, says: "progress starts at 0.2 instead of 0" }
  - { file: get_lr/mistakes/short_denominator.py, fails: decay_ends, says: "progress only reaches 0.8" }
  - { file: get_lr/mistakes/raw_cos.py, fails: decay_ends, says: "negative" }
  - { file: get_lr/mistakes/no_pi.py, fails: decay_ends, says: "only turns through 1 radian" }
  - { file: get_lr/mistakes/linear.py, fails: cosine_shape, says: "Your decay is a straight line" }
  - { file: get_lr/mistakes/hard_coded.py, fails: follows_config, says: "a number is typed in" }
```

### Clip the gradients

```challenge
id: clip
file: train.py
minutes: 25
start: { from: get_lr }
setup:
  - say: Now a model whose gradients blew up. The loss is multiplied by 20 to fake one terrible batch.
    type_file: clip/stub.py
    pause: 2.6
  - say: "Your turn: measure all the gradients as one vector, and if it's too long, shrink every gradient by the same factor."
    pause: 1.2
tests: clip/tests.py
solution: clip/solution.py
alternatives:
  - clip/alternatives/tensor_clamp.py
  - clip/alternatives/reassign_grad.py
  - clip/alternatives/norms_of_norms.py
  - clip/alternatives/if_branch.py
hints:
  start:
    try: |
      p = params[0]
      print(tuple(p.grad.shape))
      print((p.grad ** 2).sum() ** 0.5)
    see: |
      (16, 4)
      tensor(172.8638)
    next: "That's the length of the first parameter's gradient on its own. `total_norm` is the same measure over all the parameters laid end to end, so the squares of every parameter get added up before the root. Then `scale` shrinks everything when that length is above `max_norm`, and leaves it alone otherwise."
  logic: |
    total_norm = length of ALL the gradients
      laid end to end as one long vector
    factor = max_norm ÷ total_norm
      (add a tiny amount first, so it never divides by 0)
    factor above 1?   (the gradient is already short)
      yes → use 1 instead, so nothing grows
      no  → keep the factor
    for each parameter:
      multiply its gradient by the factor, in place
    → return total_norm   (measured before clipping)
  tools: |
    length of all gradients as one vector
      t ** 2              squares every value
      t.sum()             adds up all the values in t
      sum(items)          adds up a sequence of numbers
      torch.sqrt(t)       square root
    factor, never above 1
      min(a, b)           the smaller of a and b
      float(t)            a one-value tensor as a plain number
      plain arithmetic: / +
    scale each gradient in place
      t.mul_(x)           multiplies t by x, in place
  nudge: Imagine laying every gradient end to end in one long vector. How long is it? And what single number shrinks that length to max_norm, without ever making it longer?
  concept: "The global norm is the square root of the sum of every gradient entry squared, across all parameters. The factor is max_norm / total_norm, capped at 1 so short gradients stay as they are. Multiplying every p.grad by that same factor keeps the update's direction and only shortens it."
  example: |
    Trace it with two tiny gradients and `max_norm = 1.0`: `p1.grad = [3, 0]` and `p2.grad = [0, 4]`:

    - Squares added up: 9 + 16 = 25. So `total_norm = 5`.
    - The factor is `1 / 5` ≈ `0.2`, below 1, so it's used.
    - Scaled gradients: `[0.6, 0]` and `[0, 0.8]`. Their new norm is √(0.36 + 0.64) = `1.0`.
    - The direction of the update hasn't changed (3 : 4 is still 0.6 : 0.8). It's just shorter.

    Now a calm batch: `total_norm = 0.5`. The factor `1 / 0.5 = 2` is capped at 1.0, so nothing changes.
  skeleton: |
    total_norm = torch.____(sum((p.grad ** _).sum() for p in ______))
    scale = ___(1.0, max_norm / (float(__________) + 1e-6))
    p.grad.____(scale)
  solution: |
    total_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in params))
    scale = min(1.0, max_norm / (float(total_norm) + 1e-6))
    p.grad.mul_(scale)
on_solve: "Your clip_grad_norm does what torch.nn.utils.clip_grad_norm_ does, down to the 1e-6. Look at the norm before clipping: over 400. Logging that number every step is how you spot the bad batch in the recording below."
mistakes:
  - { file: clip/mistakes/placeholders.py, fails: returns_norm, says: "`total_norm` is still the placeholder" }
  - { file: clip/mistakes/summed_norms.py, fails: returns_norm, says: "Lengths don't add like that" }
  - { file: clip/mistakes/no_sqrt.py, fails: returns_norm, says: "the sum of squares" }
  - { file: clip/mistakes/last_only.py, fails: returns_norm, says: "the last tensor's norm alone" }
  - { file: clip/mistakes/returns_after.py, fails: clips_to_max, says: "the norm *after* clipping" }
  - { file: clip/mistakes/loop_noop.py, fails: clips_to_max, says: "nothing changed them" }
  - { file: clip/mistakes/rebinds_p.py, fails: clips_to_max, says: "Rebinding the loop variable `p`" }
  - { file: clip/mistakes/inverted.py, fails: clips_to_max, says: "grew to norm 25.00" }
  - { file: clip/mistakes/per_tensor.py, fails: clips_to_max, says: "one shared factor, computed from the global norm" }
  - { file: clip/mistakes/clamp_values.py, fails: clips_to_max, says: "You clamped each value" }
  - { file: clip/mistakes/always_scales.py, fails: small_untouched, says: "Clipping only ever shrinks" }
```

### Retrain Shakespeare with both

```replay
id: schedule_and_clip
intro: "Rerun 1.2's Shakespeare model with warmup, cosine decay and clipping at 1.0. Same model, same 5,000 steps. Here's what a healthy run looks like."
log: replays/schedule-and-clip.jsonl
speed: 40
recorded:
  hardware: RTX 3090
  illustrative: true
  note: Sample data written for the course to show what clipping and the schedule do. Not a real recording. Replace it with a log of your own run.
chart:
  title: Gradient norm before clipping, every 250 steps
  flag_above: 1.0
  x_labels: [step 0, step 5000]
  caption: Red bars were clipped down to 1.0. The first steps, while the model is still random, and one bad batch at step 2750. The loss barely notices it.
summary:
  - { label: Val loss, value: 1.52 → 1.47 }
  - { label: Change, value: −3.3% }
  - { label: Throughput, value: "310k tok/s" }
run_yourself: |
  # on your machine or Colab, from the course repo
  python train_gpt.py --config shakespeare_char --max-iters 5000 \
    --lr 1e-3 --min-lr 1e-4 --warmup 100 --clip 1.0
```

### Record your metrics

```metrics
id: record
intro: Your 1.2 val loss next to this run's, and this run's throughput. The change is compared with the reference run; anything that doesn't make val loss worse passes.
```

## Hints

- `torch.optim.lr_scheduler` exists, but writing the schedule as a plain function of the step keeps it obvious and lets you resume a run at any step. Set it each step with `for g in optimizer.param_groups: g["lr"] = get_lr(step)`.
- Log the grad norm that `clip_grad_norm_` returns every step. A norm that climbs for hundreds of steps means the learning rate is too high; a single spike is usually one odd batch.
- The floor matters. Decaying all the way to 0 wastes the last steps, which can't move the weights any more.
- On the GPU, clip after `loss.backward()` and before `optimizer.step()`. With mixed precision, unscale the gradients first, or you'll clip the scaled ones.

## Done when

- [ ] get_lr warms up, then follows a cosine down to min_lr {#get_lr auto=challenge:get_lr}
- [ ] clip_grad_norm only ever shrinks, and keeps the direction {#clip auto=challenge:clip}
- [ ] Val loss change and throughput recorded {#metrics_recorded auto=metrics}
