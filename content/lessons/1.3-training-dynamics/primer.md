---
gist: Training goes better when the step size starts small, peaks, then glides down along a cosine curve, and when a step whose gradients are suspiciously long is shortened before it's taken.
minutes: 16
---

## What this lesson is for {#why}
> Same model, same data, a better result: a learning-rate schedule and gradient clipping take 1.2's Shakespeare model from 1.52 to about 1.47.

In 1.2 you trained with a constant learning rate. That's a compromise. Early on the model is fragile and big steps can knock it somewhere it never recovers from. Late in training it needs small steps to settle into a good spot. One fixed number can't be right for both.

This lesson adds two small pieces of code that every serious training run uses. A **learning-rate schedule** changes the step size over the run, and **gradient clipping** shortens any single step whose gradient is suspiciously large. Neither adds a parameter or changes the model. 1.4's TinyStories run uses both, so you'll be glad to have them when a 45-minute run is at stake.

## What you need from before {#before}
> Gradients, the update step and the training loop from phase 0.

- **Gradient**: for every parameter, which direction increases the loss and how strongly. `loss.backward()` fills each parameter's `.grad`.
- **Update**: move each parameter a little *against* its gradient. The **learning rate** (lr) sets how far.
- **Optimizer**: the object that does the update. These runs use **AdamW**, which adapts each parameter's step using running averages of its recent gradients. You don't need its details here, only that `optimizer.step()` applies the update with the lr stored in `optimizer.param_groups`.
- **One training step**: forward pass, loss, `backward()`, `step()`, then clear the gradients.

## The learning rate is a step size {#lr}
> Too large and training overshoots and bounces around; too small and it crawls. The best value changes during a run.

Picture walking downhill in fog with a fixed stride. Long strides cover ground fast but can step clean over the valley and up the other side. Short strides never overshoot but take all day. Early on, while you have no idea where the valley is, cautious steps are safer. Mid-run you want to move. Near the bottom you want small steps so you settle instead of pacing back and forth.

A schedule is a function from the step number to the learning rate. You write it as a plain Python function, `get_lr(step)`, and the training loop sets the optimizer's lr from it every step.

## Warmup {#warmup}
> For the first few steps the learning rate climbs in a straight line from almost nothing to its peak.

At step 0 the weights are random and AdamW's running averages are built from a single batch, so its sense of how big each gradient "usually" is is unreliable. Full-size steps at that moment can throw the weights into a bad region. **Warmup** keeps the first steps small while those estimates settle.

With the challenge's settings, peak `max_lr = 6e-4` and 10 warmup steps:

| step | lr |
|---|---|
| 0 | 6.0e−5 |
| 4 | 3.0e−4 |
| 9 | 6.0e−4 |

The lr climbs by the same amount every step and reaches the peak on the last warmup step. (Starting from exactly 0 at step 0 is also fine, and the tests accept it, but it wastes a step.) Writing `6e-4` means 6 × 10⁻⁴, or 0.0006.

## Cosine decay {#cosine}
> After warmup, the learning rate glides from the peak down to a floor along half a cosine curve: slowly at first, fast in the middle, slowly again at the end.

First measure how far through the decay you are, as a number called **progress** that goes from 0 at the end of warmup to 1 at `max_steps`. With warmup ending at step 10 and the decay ending at step 50, the decay is 40 steps long, so step 30 is progress 0.5.

Then turn progress into a factor that slides from 1 to 0:

| progress | cos(π·p) | ½(1 + cos) |
|---|---|---|
| 0 | 1 | 1 |
| 0.25 | 0.71 | 0.85 |
| 0.5 | 0 | 0.5 |
| 0.75 | −0.71 | 0.15 |
| 1 | −1 | 0 |

cos(π·p) runs from 1 down to −1 over half a turn. Adding 1 gives 2 → 0, and halving gives 1 → 0. Finally, the lr is the floor plus that factor times the gap between peak and floor. At progress 0.5 it's exactly halfway: (6e−4 + 6e−5) / 2 = 3.3e−4, which is what the Learn deck asks you to predict for step 30.

Compared with a straight line, the cosine spends longer near the peak, where learning is fast, and longer near the floor, where the model settles.

```chart
kind: line
title: The challenge's schedule, cosine against a straight line
x: { label: step, values: [0, 4, 9, 20, 30, 40, 50] }
y: { label: learning rate, min: 0, max: 0.0007 }
series:
  - { name: cosine, values: [0.00006, 0.0003, 0.0006, 0.000519, 0.00033, 0.000141, 0.00006] }
  - { name: straight line, values: [0.00006, 0.0003, 0.0006, 0.000465, 0.00033, 0.000195, 0.00006] }
highlight: { x: 30, label: "halfway, 3.3e−4" }
caption: Both warm up to the 6e−4 peak by step 9 and meet at step 30, but the cosine stays nearer the peak early on and nearer the 6e−5 floor late.
```

## The floor {#floor}
> The decay ends at a minimum learning rate, here a tenth of the peak, and holds there after max_steps.

Decaying all the way to 0 would make the last steps useless: they couldn't move the weights at all. A floor of `min_lr = max_lr × 0.1` keeps them doing a little work. After `max_steps` the schedule just returns the floor, in case training runs a few steps longer than planned.

## Schedule mistakes and their symptoms {#schedule-mistakes}
> Each mistake shows up at a specific step: step 0, the step after warmup, max_steps, or after it.

| Mistake | What the tests see |
|---|---|
| no warmup | step 0 already at the peak |
| warmup not scaled by max_lr | lr reaches 1.0 at step 9 |
| 0 after max_steps | training stops learning |
| progress = step / max_steps | lr drops at once after warmup |
| divide by max_steps only | never reaches the floor |
| cos without ½(1 + …) | lr goes negative |
| cos without π | turns only 1 radian |
| straight-line decay | wrong shape in the middle |
| typed-in numbers | breaks for another config |

The last one is caught by rerunning your function with different settings: peak 1e−3, floor 1e−4, 20 warmup steps, 100 total. A schedule that reads its four settings works unchanged.

## Gradient norm: how long is a gradient? {#norm}
> Lay every gradient end to end as one long vector. Its length, the square root of the sum of all the squares, is the global gradient norm.

The length of a vector is √(sum of squares), like the 3-4-5 triangle: `[3, 4]` has length √(9 + 16) = 5.

A model has many parameter tensors, each with its own gradient. Clipping looks at all of them together:

```
p1.grad = [2, -1, 2]
  squares: 4 + 1 + 4 = 9
p2.grad = [[0, 4]]
  squares: 0 + 16 = 16

global norm = √(9 + 16) = 5
```

Each tensor alone has norm 3 and 4, but the global norm is 5, not 7. Lengths don't add; squares do. The tests check for exactly these numbers: 7 (norms added), 25 (no square root) and 4 (only the last tensor, because the loop replaced the total instead of adding to it).

## Clipping by the global norm {#clip}
> If the global norm is over the limit, multiply every gradient by the same factor, limit ÷ norm. If it's under, leave everything alone.

With a limit of 1.0 and a norm of 5, the factor is 1 / 5 = 0.2:

```
p1 [2, -1, 2] → [0.4, -0.2, 0.4]
p2 [0, 4]     → [0.0, 0.8]
new global norm = 1.0
```

Every gradient shrinks by the same factor, so the update still points in exactly the same direction. It's just shorter. Two tempting alternatives change the direction:

| Approach | Result | Problem |
|---|---|---|
| one shared factor | norm 1.0 | none |
| clip each tensor to 1 | norm 1.41 | direction bends |
| clamp each value to ±1 | `[1,-1,1]`, `[0,1]` | direction bends |
| always multiply by limit ÷ norm | norm 5 → 10 when limit is 10 | grows small gradients |

The factor is capped at 1 because clipping only ever shrinks. PyTorch also adds a tiny 1e−6 to the norm before dividing, so a norm of exactly 0 can't divide by zero.

## Where clipping goes in a step {#where}
> After backward() has filled the gradients, and before optimizer.step() uses them.

Clipping reads gradients, so they have to exist: after `loss.backward()`. It changes them, so it has to happen before `optimizer.step()` consumes them. On the GPU you'll call `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`, which does what your function does and **returns the norm before clipping**. Log that number every step. A norm that climbs steadily means the learning rate is too high; one isolated spike is usually a single odd batch. Your function returns the norm before clipping too, for the same reason.

## Reading the recording {#run}
> 1.2's model again, now with 100 warmup steps, cosine decay from 1e−3 to 1e−4 and clipping at 1.0. Val loss ends near 1.47.

The recording is **illustrative**: sample data written for the course to show what clipping and the schedule do, not a real recording. Its chart shows the gradient norm before clipping every 250 steps. Red bars went over 1.0 and were clipped:

- **Step 0 and 250**: the model is still near random and the norms are large. Clipping keeps those early steps tame.
- **Step 2,750**: one bad batch spikes the norm to 3.41. It gets clipped to 1.0 and the loss barely notices.
- **The rest**: norms settle around 0.3, well under the limit, so clipping does nothing.

```chart
kind: line
title: Training loss in the recording
x: { label: step, values: [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000] }
y: { label: training loss, min: 0, max: 4.5 }
series:
  - { name: training loss, values: [4.29, 2.31, 1.98, 1.72, 1.60, 1.52, 1.46, 1.42, 1.38, 1.35, 1.32, 1.31, 1.28, 1.26, 1.24, 1.23, 1.21, 1.20, 1.19, 1.18, 1.18] }
highlight: { x: 2750, label: "clipped spike, loss barely moves" }
illustrative: true
caption: The loss falls fast while the model is near random, then flattens as the learning rate glides down to its floor.
```

Throughput is printed with each eval. Each step processes 64 × 256 = 16,384 tokens, and at about 53 ms per step that's roughly 310,000 tokens per second on an RTX 3090. A laptop CPU manages a few thousand.

## Your metrics {#metrics}
> The quality metric is the percentage change in val loss from 1.2's constant learning rate to this run. Negative is better.

Percent change is (new − old) / old × 100. With the recording's numbers, (1.47 − 1.52) / 1.52 × 100 ≈ −3.3%. It passes as long as the val loss didn't get worse. The systems metric is your throughput in tokens per second. The pass mark of 20,000 assumes a GPU; if you trained on a CPU, record your number anyway.

## Words you'll see {#glossary}
> The terms in this lesson's cards, challenges and metrics.

- **Learning rate (lr)**: how far each update moves the parameters.
- **Schedule**: a function from the step number to the learning rate.
- **Warmup**: the first steps, where the lr climbs in a straight line to its peak.
- **max_lr, min_lr**: the peak and the floor of the schedule.
- **warmup_steps, max_steps**: where warmup ends and where the decay ends.
- **Cosine decay**: lowering the lr along half a cosine curve, from the peak to the floor.
- **Progress**: how far through the decay a step is, from 0 to 1.
- **AdamW**: the optimizer used here. It adapts each parameter's step and decays weights separately.
- **param_groups**: the optimizer's settings, including the lr you set every step.
- **Gradient norm**: the length of the gradients laid end to end, √(sum of all squares).
- **Global norm**: that length over every parameter tensor together.
- **Gradient clipping**: scaling all gradients down by one factor when their global norm is over a limit (max_norm).
- **clip_grad_norm_**: PyTorch's clipping function. It returns the norm before clipping.
- **Loss spike**: a sudden jump in loss, often from one bad batch.
- **Throughput (tok/s)**: training tokens processed per second.
- **Percent change**: (new − old) / old × 100.
