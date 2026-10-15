---
paper: LoRA
gist: Freeze the pretrained weights and learn each layer's update as a product of two thin matrices. You train a tiny fraction of the parameters, match full fine-tuning, and add no inference cost once merged.
---

## The problem {#problem}
> Full fine-tuning gives you a full copy of a huge model for every task, and its optimizer state costs several times the weights' memory.

Fine-tuning GPT-3 175B the usual way means storing and serving a separate 175B-parameter model per task, and training it needs gradients and Adam state for every weight. Earlier cheap alternatives had their own costs. Adapter layers add extra sequential computation, so inference gets slower. Prefix tuning eats into the usable sequence length and is hard to optimize. The authors wanted something cheap to train that costs nothing extra at inference.

## The low-rank hypothesis {#hypothesis}
> The change fine-tuning makes to a weight matrix has a low "intrinsic rank", so it can be written as a product of two small matrices.

Earlier work had shown that pretrained models can be adapted inside a surprisingly small subspace. LoRA assumes the same of the update itself: ΔW doesn't need all in × out degrees of freedom. If ΔW = B A with A of shape [r, in] and B of shape [out, r], it has only r × (in + out) numbers, and r can be very small.

## The method {#method}
> Keep W frozen, compute W x + (alpha / r) · B A x, start B at zero so training begins exactly at the pretrained model, and train only A and B.

A starts as random Gaussian values and B as zeros, so B A is zero and the first forward pass equals the base model's. The update is scaled by alpha / r. The paper's reason is practical: with that scaling you can change r without retuning the learning rate much. Only A and B get gradients and optimizer state. This is exactly the layer you write in 4A.1.

## No extra inference latency {#merge}
> After training you can add (alpha / r) · B A into W, so the deployed model has the original shapes and the original speed.

Because the update is linear and sits beside W, merging is a single addition per layer. Switching tasks means subtracting one adapter's B A and adding another's. Adapters inserted between layers can't do this, which is where their extra latency comes from.

## Where to put it, and how small r can be {#where}
> On GPT-3 they adapted only the query and value projections, and ranks as low as 1 to 4 already worked well there.

With a fixed budget of trainable parameters, spreading it over several projection types at low rank beat putting it all on one projection at high rank. Later work, QLoRA among it, found that adapting every Linear layer works better still. That's the setting 4A.1's replay and 4A.2 use.

## What it achieves {#results}
> On RoBERTa, DeBERTa, GPT-2 and GPT-3, LoRA matched or beat full fine-tuning while training a tiny fraction of the parameters.

For GPT-3 175B the paper reports about 10,000 times fewer trainable parameters and roughly a third of the training GPU memory, because the frozen weights need no optimizer state. Checkpoints for each task shrink from hundreds of gigabytes to a few megabytes.

## What the update looks like {#analysis}
> The learned ΔW mostly amplifies directions that were already in W but weakly expressed, rather than copying W's strongest directions.

The authors compared the subspace ΔW spans with W's singular directions. ΔW correlated more with W than a random matrix would, but not with W's top directions. Their reading is that adaptation boosts features the pretrained model already had but didn't stress. It's a hint about why so few directions are enough.

## How the course uses it {#course}
> 4A.1 builds the LoRA layer and its training step by hand; 4A.2 to 4A.5 use the same idea on a 4-bit Llama 3.1 8B.

Your `LoRALinear` follows the paper's layout: A is [r, in], B is [out, r], B starts at zero, and the scale is alpha / r. The browser task even shows the low-rank point directly: a rank-4 adapter fits a rank-2 change exactly, and a rank-1 adapter can't.
