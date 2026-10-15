---
gist: Continued pretraining keeps training Llama on plain Mongolian text so its 8,000 new rows learn what they mean. To compare it fairly with the control run, you train on exactly the same documents, plan the steps and hours, and set the learning rate's schedule.
minutes: 17
---

## What this lesson is for {#why}
> The extended model has 8,000 untrained rows. This lesson trains them, and the rest of Llama through LoRA, on the same Mongolian text the control run used, then checks whether the new tokenizer beat it.

4B.2 left you with a model that starts about 4% worse per character than Llama 3.1, because its new rows have never seen a training example. **Continued pretraining** fixes that: keep doing what pretraining did, predicting the next token, on Mongolian text.

This is the step where 4B's research question gets its first real answer. 4A.4's **control run** trained the same model on the same text with Llama's own tokenizer. If the extended model ends lower in bits per character on the same held-out text, the new tokenizer helped.

Before spending GPU hours you plan the run in the browser: how many tokens the control's text becomes, how many steps and hours that takes, and how the learning rate changes over those steps.

## What you need from before {#before}
> From 4A: LoRA trains small add-on matrices instead of the whole model, QLoRA keeps the frozen model in 4 bits, and bits per character compares models across tokenizers.

The pieces you'll reuse:

- **LoRA**: instead of changing a big weight matrix, train two thin matrices whose product is added to it. 4A.4 used rank 32 on all 7 linear layers in all 32 blocks.
- **QLoRA**: the frozen base model is stored in 4-bit NF4 so it fits a 12 GB card.
- **Step**: one optimizer update, computed from one batch. 4A.4's batch is 4 sequences of 2,048 tokens, 8,192 tokens per step.
- **Epoch**: one pass over the training text.
- **Bits per character**: total loss in nats, divided by ln 2, divided by characters. It doesn't depend on how a tokenizer cut the text.

The control run, from 4A.4's log:

| | Control (4A.4) |
|---|---|
| Text | a fixed 20M-token sample, 1 epoch |
| Steps | 2,441 |
| Learning rate | 1e-4, LoRA rank 32 |
| Held-out bits per char | 1.04 → 0.67 |
| Time | 7.1 GPU hours on an RTX 3060 12 GB |

## Same documents, not same tokens {#same-text}
> A fair comparison trains on the same documents. With the extended tokenizer those documents are fewer tokens, so the run is shorter.

It's tempting to train on "20M tokens" again. But 20M extended tokens hold about twice as much text, because each one covers more. A win could then come from reading more Mongolian, not from the tokenizer.

So you train on the control's exact documents. 4A.4 saved their ids in `sample_ids.txt`. Count them in extended tokens by going through words:

```
20M Llama tokens / 3.38
  = 5.92M words
5.92M words × 1.62
  = 9.59M extended tokens
```

3.38 and 1.62 are tokens per word, from 1.6 and 4B.1. The text is the same, so the number of words is the same. Only the cost per word changes.

## Steps and hours {#plan}
> Steps are tokens divided by tokens per step, as a whole number. Hours are tokens divided by tokens per second, divided by 3,600.

Keep the control's batch, so both runs take updates of the same size:

| | Control | Extended |
|---|---|---|
| Tokens | 20.0M | 9.59M |
| Tokens per step | 8,192 | 8,192 |
| Steps | 2,441 | 1,170 |
| Tokens per second | 790 | 760 |
| Training hours | 7.0 | 3.5 |

The extended run is slightly slower per token (760 against 790) because its new rows train and its LM head is bigger. It still finishes in about half the time, because there are half as many tokens.

Mistakes the tests check for: steps with a fraction (an optimizer step is a whole batch), multiplying where you should divide, and stopping at seconds or minutes instead of hours.

## The learning rate and its schedule {#lr}
> The learning rate sets how big each update is. It starts small, climbs to a peak during warmup, then falls along half a cosine to a floor.

The learning rate is the step size: how far each update moves the weights. A fixed rate is rarely best. This run uses three phases:

```
lr
1e-4 |    ____
     |   /    ‾‾‾--__
     |  /            ‾--__
1e-5 | /                  ‾‾‾‾‾
     +------------------------- step
      0  50        610      1170
       warm   cosine decay   floor
```

- **Warmup**, steps 0 to 49: a straight climb to the peak, 1e-4, the same peak as the control.
- **Cosine decay**, steps 50 to 1,169: half a cosine wave from the peak down to the floor. It stays near the top for a while, drops fastest in the middle, and flattens out at the end.
- **Floor**, from step 1,170 on: stay at the minimum, 1e-5, which is 10% of the peak.

Numbers you can check: step 0 is 1e-4 × 1/50 = 2e-6, step 25 is 1e-4 × 26/50 = 5.2e-5, and step 50 is the peak. Halfway through the decay, the rate is halfway between peak and floor: (1e-4 + 1e-5) / 2 = 5.5e-5.

Mistakes the tests check for: decaying to 0 instead of the floor, starting the decay at step 0 instead of after warmup, forgetting π (the curve barely moves), using a full cosine (it comes back up), a straight line instead of a curve, an upside-down curve, and dropping to 0 after the end.

One caveat. 4A.4's command doesn't name a schedule, so check what your control actually ran: its log prints the learning rate. The two runs need the same schedule, or the comparison measures two changes at once.

## Why warm up {#warmup}
> Adam scales each update by a running average of past gradients. At the start that average is empty, and fresh rows have large gradients, so full-size first steps can knock pretrained weights off course.

Adam, the optimizer used throughout the course, divides each weight's update by an estimate of how large that weight's gradients usually are. That estimate needs a few dozen steps to settle. Meanwhile the 8,000 new rows produce large gradients, because they start far from where they should be. Small early steps give both time to settle before the rate reaches its peak.

## The new rows' own learning rate {#new-rows}
> The new rows train in full with their own, higher learning rate. Keep them as a separate 8,000-row parameter so you don't pay memory for the 128,256 frozen rows.

LoRA adapts the transformer. The new rows of `embed_tokens` and `lm_head` are trained directly, because they have to be learned from scratch. They usually want a higher rate than LoRA, about 5× (5e-4 against 1e-4), set in their own optimizer parameter group.

How you make only those rows trainable matters on a 12 GB card:

| Approach | Memory for training state |
|---|---|
| Separate 8,000-row parameter | 65.5M parameters, about 0.8 GB |
| Whole matrices, gradient hook zeroes old rows | 1.1B parameters, about 13.4 GB |

The hook looks simpler, but it still keeps gradients and Adam's two moments (12 bytes per parameter in fp32) for every row of both matrices. Weight decay would also move the "frozen" rows.

## Forgetting and data mixing {#forgetting}
> Training only on Mongolian slowly erodes what the model knew about everything else. Mixing in some general text guards against it, but only if the control does the same.

When a model trains on one kind of text, its weights drift away from what served other text. That's called **catastrophic forgetting**. The standard guard is **replay**: mix some general data, for example English web text, into every batch.

This run adds no English, because the control added none. If one run mixes in English and the other doesn't, the comparison measures two changes. Instead both runs check English perplexity on the same English sample:

| | English perplexity |
|---|---|
| Base Llama 3.1 | 6.10 |
| Control after training | 6.41 (+5.1%) |
| Extended after training | 6.47 (+6.1%) |

A few percent is normal for an all-Mongolian run. If yours rises much more, look at the new-rows learning rate first. If you later train longer, add English replay to both runs.

## Reading the run {#reading}
> Watch held-out bits per character fall, and compare it with the control's 0.67. Ignore per-token perplexity across the two tokenizers.

The run evaluates every 250 steps. Healthy output looks like this (reference run, illustrative):

| Step | Bits per char |
|---|---|
| 0 | 1.08 |
| 250 | 0.86 |
| 500 | 0.76 |
| 750 | 0.69 |
| 1,000 | 0.662 (below the control) |
| 1,170 | 0.642 |

0.642 against the control's 0.67 is 4.2% better. The log also prints per-token perplexity, 9.70 against the control's 3.12. That looks far worse, but an extended token covers about 5.1 characters and a Llama token about 2.4, so per-token numbers can't be compared. This is the lesson's main trap.

Two signs of trouble: bits per character rising while training loss keeps falling means the model is memorising its training text (keep the best checkpoint, not the last), and a sharp rise in English perplexity means it's forgetting.

## What you'll run {#run}
> On the control's 12 GB GPU: about 3.6 GPU hours of continued pretraining on the control's documents, with evals every 250 steps. The recording in the lesson is illustrative sample data.

The command loads 4B.2's initialised model in 4-bit, uses the control's `sample_ids.txt`, LoRA rank 32, peak 1e-4 with the schedule above, and 5e-4 for the new rows. It saves the best checkpoint for 4B.4.

You record two metrics:

- **Loss against the control**: the percent change from your control's bits per character to your extended run's. The reference is −4.2%, and it passes at 0% or below.
- **GPU hours**: 3.6 in the reference run. It passes at 7.1 or less, the control's time.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Continued pretraining (CPT)**: more next-token training on new text, starting from a pretrained model.
- **Control run**: 4A.4's run with Llama's own tokenizer, the baseline every 4B result is compared with.
- **LoRA / QLoRA**: training small add-on matrices; QLoRA keeps the frozen model in 4 bits.
- **Step**: one optimizer update from one batch.
- **Tokens per step**: sequences per batch × sequence length, 4 × 2,048 = 8,192 here.
- **Epoch**: one pass over the training text.
- **Throughput**: tokens processed per second.
- **GPU hours**: wall-clock hours the run keeps the GPU busy.
- **Learning rate (lr)**: the size of each update.
- **Warmup**: the opening steps where the learning rate climbs from near 0 to its peak.
- **Cosine decay**: the learning rate falling along half a cosine from the peak to the floor.
- **min_lr (floor)**: the lowest learning rate, kept after the decay ends.
- **Parameter group**: a set of weights with its own optimizer settings, such as a higher learning rate.
- **Catastrophic forgetting**: losing old abilities while training on new data.
- **Replay**: mixing old or general data into training to prevent forgetting.
- **Bits per character**: loss per character in bits, comparable across tokenizers.
- **Perplexity**: e to the power of the loss per token. Only comparable within one tokenizer.
- **Overfitting**: training loss falling while held-out loss rises.
