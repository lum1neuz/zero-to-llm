---
gist: Fine-tuning normally means updating every weight of a model. LoRA freezes them all and learns the change as two thin matrices instead, so you train about 1% of the numbers and get nearly the same result.
minutes: 16
---

## What this lesson is for {#why}
> Phase 3 made a model smaller to run. Phase 4A changes what a model knows, and LoRA is the trick that makes that affordable on one GPU.

The course project is a Llama 3.1 8B model that reads and writes Mongolian. Meta trained that model on trillions of tokens with thousands of GPUs. You have one GPU with 12 GB of memory. You can't train an 8B model from scratch, and as you'll see below, you can't even fine-tune all of its weights the ordinary way.

**LoRA** (Low-Rank Adaptation) gets around this. It leaves the pretrained weights exactly as they are and trains a small add-on beside each layer. In this lesson you build that add-on by hand in the browser, train it, and then use it on your 30M TinyStories model from phase 1. 4A.2 applies the same idea to Llama 3.1 8B.

## What you need from before {#before}
> A Linear layer computes `x @ W.T + b`, a training step is zero-grad, loss, backward, step, and Adam keeps two extra numbers for every weight it trains.

Three things from earlier lessons come back here:

- **`nn.Linear(in, out)`** stores `weight` with shape `[out, in]` and `bias` with shape `[out]`. For an input `x` of shape `[batch, in]` it returns `x @ weight.T + bias`, shape `[batch, out]`.
- **A training step** is the loop you've written since micrograd: clear the old gradients, run the model and compute the loss, call `backward()` to fill every `.grad`, then let the optimizer update the weights.
- **Adam** keeps two running averages for every parameter it trains: one of the gradient (`m`) and one of the squared gradient (`v`). That's memory on top of the weights themselves.

One more piece of PyTorch: every parameter has a flag, `requires_grad`. When it's `False`, backward doesn't compute a gradient for it and you can't train it. Turning it off is called **freezing**.

## Pretraining, fine-tuning, and what it costs {#cost}
> Pretraining teaches a model language from scratch. Fine-tuning nudges a trained model toward new data. Done the full way, the nudge needs about 16 bytes for every weight.

**Pretraining** starts from random weights and trains on a huge pile of text until the model predicts the next token well. That's what phase 1 did on TinyStories. **Fine-tuning** starts from a pretrained model and keeps training on new data: Mongolian text, or a style of story. The model already knows a lot, so the changes it needs are small.

The obvious way to fine-tune is to keep training every weight. Count the memory for each trained weight, with everything stored as fp32 as in this course's accounting:

| What | Bytes per weight |
|---|---|
| The weight itself | 4 |
| Its gradient | 4 |
| Adam's `m` | 4 |
| Adam's `v` | 4 |
| **Total** | **16** |

For Llama 3.1 8B that's 8 billion × 16 bytes, about 128 GB, before you've stored a single activation. Your GPU has 12. Even with clever tricks the full approach is out of reach. The fix is to train far fewer numbers.

## What "rank" means {#rank}
> A matrix of rank 1 is one column times one row. A rank-r matrix is a sum of r of those, which you can store as two thin matrices.

Take a column of 4 numbers and a row of 4 numbers, and multiply every pair:

```
col = [1, 2, 3, 4]
row = [1, 0, 2, 1]

col @ row  (4×1 times 1×4)
[[1, 0, 2, 1],
 [2, 0, 4, 2],
 [3, 0, 6, 3],
 [4, 0, 8, 4]]
```

That 4 × 4 matrix has 16 numbers, but you stored only 8 to make it. Every row is a multiple of `[1, 0, 2, 1]`, so it only has **one direction** in it. That's what **rank 1** means.

Add a second column-times-row and you get a **rank-2** matrix: two independent directions. Stack the columns side by side into `B` (shape `[4, 2]`) and the rows into `A` (shape `[2, 4]`), and the matrix is `B @ A`.

On a 4 × 4 matrix this saves nothing: `B` and `A` hold 8 + 8 = 16 numbers, the same as the matrix. The saving comes with size. A 4096 × 4096 matrix has 16.8 million numbers. A rank-16 version, stored as `B` `[4096, 16]` and `A` `[16, 4096]`, has 131,072. That's 0.78%.

## The LoRA layer {#lora}
> Keep the pretrained weight W frozen and add a trainable low-rank update B @ A beside it. Only A and B ever change.

The bet LoRA makes is that the *change* fine-tuning needs, call it ΔW, has low rank. So instead of training W directly, it writes ΔW = B @ A and trains only those two:

```
output = base(x) + scale * x A^T B^T
```

Here are the shapes for the browser task's layer: 6 inputs, 10 outputs, rank r = 4, batch of 3.

| Tensor | Shape | Notes |
|---|---|---|
| `x` | `[3, 6]` | the input |
| `W` (frozen) | `[10, 6]` | the pretrained weight, `[out, in]` |
| `A` | `[4, 6]` | `[r, in]`, stored like a Linear weight |
| `B` | `[10, 4]` | `[out, r]` |
| `x @ A.T` | `[3, 4]` | squeezed down to r numbers per row |
| `… @ B.T` | `[3, 10]` | expanded back out, same as `base(x)` |

`A` squeezes each input down to r numbers, and `B` expands them back to the output size. If you get a shape error in `forward`, walk through this table: `x` can only meet `A` first, and both need `.T`.

PEFT, the library you'll use on the GPU, stores these as `lora_A` `[r, in]` and `lora_B` `[out, r]`, exactly this layout.

## Why B starts at zero {#zero}
> With B all zeros, B @ A is zero, so on step 0 the layer is exactly the pretrained layer. A must be random, or nothing ever trains.

Fine-tuning should start from the model you were given, not from a damaged copy. If `B` starts as zeros, `B @ A` is a matrix of zeros whatever `A` holds, and `base(x) + 0` is `base(x)`, bit for bit. The test checks this with `torch.equal`.

Why not start both at zero? Look at where the gradients come from. `B`'s gradient is built from `x @ A.T`, and `A`'s gradient is built from `B`. If both are zero, both gradients are zero, every step. Adam can't rescue a gradient of exactly zero. The layer would never move. So `A` starts with small random values, `B` with zeros, and on the first step `B` gets a real gradient.

A classic mistake goes the other way: start `B` random. The model is then already different from the pretrained one before training begins.

## The alpha / r scale {#scale}
> The update is multiplied by alpha / r, so raising the rank doesn't make the update bigger and the learning rate you tuned keeps working.

A rank-32 `B @ A` adds up 32 directions, a rank-8 one adds up 8. Without a correction, a higher rank would also mean a bigger update. LoRA multiplies the update by **alpha / r**, where alpha is a number you pick once:

| alpha | r | scale |
|---|---|---|
| 16 | 8 | 2.0 |
| 16 | 16 | 1.0 |
| 16 | 32 | 0.5 |

The tests check the direction: it's alpha divided by r, not r divided by alpha, and not alpha alone. A common setting is alpha = 2r, so the scale is 2.

## Counting what trains {#count}
> A LoRA adapter on one Linear layer has r × (in + out) trainable numbers. It only saves memory when r is much smaller than the layer.

`A` has r × in numbers and `B` has out × r. Together that's r × (in + out).

| Layer | Full weight | LoRA, r | LoRA numbers | Share |
|---|---|---|---|---|
| Browser task, 6 → 10 | 60 (+10 bias) | 4 | 64 | 47.8% |
| Llama q_proj, 4096 → 4096 | 16,777,216 | 16 | 131,072 | 0.78% |
| Your 30M model, all 24 Linears | about 30.2M total | 8 | 393,216 | 1.29% |

The first row is a reminder: on a tiny layer LoRA saves almost nothing. The savings come from r being far smaller than the layer's width.

## Training only A and B {#training}
> Give the optimizer a list of exactly the parameters that still require grad, and keep the usual order: clear, forward, backward, step.

Two mistakes the tests look for, and why they matter:

- **Handing the optimizer everything.** `model.parameters()` includes the frozen weights. PyTorch won't update them, since they never get a gradient, but Adam still sets aside state for them. On Llama 3.1 8B that wasted state would be 64 GB. Filter to the parameters with `requires_grad` on.
- **Returning a generator.** In Python, `(p for p in …)` is a generator, which works like a Java `Iterator`: once you've walked it, it's empty. The script counts the parameters first and then hands them to Adam, so Adam would get nothing. Make it a list.

The training step itself is unchanged from earlier lessons. The tests also check the order. `backward()` *adds* into `.grad`, it never replaces it, so gradients must be cleared before each backward, not after it. The loss is the mean squared error, not the sum, so the learning rate doesn't depend on batch size.

The browser target is the base layer plus a rank-2 change. A rank-4 adapter can hold it exactly, so the loss drops to about zero. Try r = 1 afterwards: one direction can't hold a rank-2 change, and the loss stalls.

## Merging, and your 30M run {#merge}
> After training you can fold B @ A into W, so the model costs nothing extra to run. The GPU step does this on your 30M model.

Because the update is just added to `W`, you can compute `W + scale · B @ A` once and use that as the new weight. The merged model has the original shapes and the original speed, and you can throw `A` and `B` away.

On your GPU, the lesson wraps every Linear in your 30M TinyStories model with your LoRA layer (r = 8, alpha = 16) and fine-tunes it on **TinyStories-Instruct**, stories that come with a prompt listing features, words and a summary. The replay is an **illustrative** recording written for the course, not a real run. Healthy looks like this:

- About 1.29% of parameters trainable.
- Validation loss falling from about 2.31 at step 0 (the base model has never seen these prompts) to about 1.86, with most of the drop in the first 1,000 steps.
- A full fine-tune of the same model reaches about 1.79. LoRA gets close with a tiny fraction of the trainable numbers.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Pretraining**: training a model from random weights on a large amount of text.
- **Fine-tuning**: continuing to train a pretrained model on new data.
- **Frozen**: a parameter with `requires_grad` off. It gets no gradient and never changes.
- **LoRA (Low-Rank Adaptation)**: freezing W and training a low-rank update B @ A beside it.
- **Rank**: how many independent directions a matrix has. A rank-r matrix can be written as `[out, r] @ [r, in]`.
- **A, B**: LoRA's two thin matrices. A is `[r, in]` and starts small and random, B is `[out, r]` and starts at zero.
- **r**: the rank of the adapter, usually 8 to 64.
- **alpha**: a number you choose. The update is scaled by alpha / r.
- **Trainable parameters / trainable%**: the numbers the optimizer updates, and their share of all parameters.
- **Optimizer state**: what Adam keeps per trained parameter, the running averages `m` and `v`.
- **Merge**: adding scale · B @ A into W after training, so no adapter is needed at inference.
- **PEFT**: the Hugging Face library for LoRA and similar methods. It calls the matrices `lora_A` and `lora_B`.
- **nats**: the unit of the loss when it's computed with the natural log, as `cross_entropy` does.
- **Validation loss**: the loss on data held out from training, the lesson's quality metric.
- **TinyStories-Instruct**: TinyStories with a prompt (features, words, a summary) before each story.
