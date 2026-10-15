---
gist: Keep training Llama 3.1 8B on your Mongolian corpus with nothing else changed, so phase 4B has a fair baseline. Measure the result with perplexity pooled over all tokens, plus bits per character, which still works when the tokenizer changes.
minutes: 14
---

## What this lesson is for {#why}
> You'll run the experiment every 4B run is compared against, and write the evaluation that makes that comparison fair.

Phase 4B asks one question: does a tokenizer built for Mongolian make the model better at Mongolian? To answer it, you need something to compare against. Training on Mongolian text helps a lot no matter which tokenizer you use, so comparing a 4B model with untouched base Llama would give the new tokenizer credit for everything the training did.

This lesson runs the **control**: the same kind of training 4B will do, on the same data, with the same settings, and with Llama's original tokenizer left alone. First you write the evaluation in the browser. Then you run the control on your GPU for about 7 hours.

## What you need from before {#before}
> The loss is the average negative log-probability of each next token, perplexity is e to that power, and QLoRA plus your clean corpus are ready from 4A.2 and 4A.3.

- **Next-token loss** (phase 1). A language model gives a probability for every possible next token. The loss at one position is −ln(probability of the token that actually came next), in **nats**. `F.cross_entropy` computes it from logits.
- **Perplexity**. e to the power of the average loss. A perplexity of 3 means the model is, on average, as unsure as if it were choosing among 3 equally likely tokens. Lower is better.
- **QLoRA** (4A.2). A 4-bit frozen base with trainable LoRA adapters fits in 12 GB.
- **The corpus** (4A.3). 191M clean tokens, with 2,000 documents held out *after* dedup.

## Continued pretraining {#cpt}
> Continued pretraining is the same next-token training as pretraining, just started from a trained model and fed new text.

Pretraining taught Llama general language from trillions of tokens, mostly English. **Continued pretraining** keeps the same objective, predicting the next token, and feeds the model a new kind of text, here Mongolian. Nothing about the task changes: no questions, no answers, just text. It's different from 4A.5's instruction tuning, which teaches the model to respond to requests.

The control does this with QLoRA: r = 32 adapters on all seven Linear layers in each block, 83.9M trainable numbers. The tokenizer, the embeddings and the output head are all frozen.

## What a control run is {#control}
> Change one thing, keep everything else identical, and the difference between two runs belongs to that one thing. It's an A/B test for training.

If you've run an A/B test on a web page, you know the idea: show version A and version B to comparable visitors at the same time and change only the thing you're testing. If B also had a different colour scheme, you couldn't tell which change caused the result.

For training, "everything else" means:

| Must match across runs | In the control |
|---|---|
| the training text | a fixed 20M-token sample, seed 0, ids saved |
| the number of steps | 2,441 |
| the adapters | r = 32 on all seven Linears |
| learning rate, sequence length | 1e-4, 2048 |
| the held-out set | 4A.3's 2,000 documents |

4B's runs change the tokenizer and keep this table. That's why the lesson asks you to save the sample's document ids.

## The loss of one document, shifted by one {#shift}
> Position t predicts the token at t + 1, so a document of n tokens makes n − 1 predictions. The first token is never predicted and the last position predicts nothing.

Say a document's ids are `[BOS, a, b, c]`. **BOS** is the special token that starts every sequence. The model returns one row of logits per position:

| Position t | Input token | Its logits predict |
|---|---|---|
| 0 | BOS | a |
| 1 | a | b |
| 2 | b | c |
| 3 | c | (nothing, the document ended) |

So you score logit rows 0 to n − 2 against tokens 1 to n − 1. Two classic mistakes:

- **No shift.** Scoring row t against token t asks the model to predict the token it was just given. A model that copies its input would look perfect.
- **Shifted the wrong way.** Scoring row t + 1 against token t asks for the token *before*.

For the next step, add up the losses of a document's predictions rather than averaging them. You'll see why next.

## Perplexity, pooled over every token {#pooled}
> Add up all the loss and all the predicted tokens across documents, then divide once. Averaging per-document perplexities lets a short document count as much as a long one.

Two held-out documents:

| Document | Predicted tokens | Total loss | Its own perplexity |
|---|---|---|---|
| A | 10 | 10 nats | e¹ ≈ 2.72 |
| B | 1000 | 2000 nats | e² ≈ 7.39 |

Pooled, every token counts once:

```
(10 + 2000) / (10 + 1000)
  ≈ 1.99 nats per token
e^1.99 ≈ 7.32
```

Averaging the two perplexities instead gives 5.05, because the 10-token document counts as much as the 1000-token one. Averaging each document's mean loss gives e^1.5 ≈ 4.48, for the same reason. The tests check for both, and for counting BOS as a predicted token. A document of n ids has n − 1 predictions.

## Why perplexity can't compare tokenizers {#tokenizers}
> Per-token perplexity depends on how much text each token covers. Bits per character divides by the text itself, so it compares runs with different tokenizers fairly.

A tokenizer built for Mongolian cuts the same sentence into fewer, longer tokens. Each token carries more text, so each one is harder to guess, and per-token perplexity goes **up**, even if the model understands the text better. In the browser task, the same three sentences give:

| Run | Per-token perplexity | Bits per character |
|---|---|---|
| control, Llama's tokenizer | 3.34 | 0.73 |
| 4B preview, a Mongolian tokenizer | 13.64 | 0.69 |

Perplexity says the preview is four times worse. Per character, it's slightly better.

**Bits per character** measures the total surprise against something both runs share, the number of characters:

```
bits = nats / ln 2
     (ln 2 ≈ 0.693)
bits per char
  = total nats / ln 2
    / characters
```

For example, 1,000 nats of loss over 2,000 characters is 1000 / 0.693 / 2000 ≈ 0.72 bits per character. The tests catch the near misses: dividing by tokens instead of characters, forgetting to convert nats to bits, and counting bytes. A Cyrillic letter is 2 bytes in UTF-8, so bits per byte comes out about half.

## The control run on your GPU {#run}
> About 7 hours of QLoRA continued pretraining on 20M tokens. Held-out perplexity should fall from about 5.8 to about 3.1, and bits per character from about 1.04 to 0.67.

You run continued pretraining for one epoch over the 20M-token sample, 2,441 steps of 4 × 2048 tokens, evaluating the held-out set every 250 steps. The replay is an **illustrative** recording written for the course, not a real run. Healthy looks like this:

| Eval point | Held-out perplexity | Bits per character |
|---|---|---|
| step 0 (base model) | 5.84 | 1.04 |
| step 500 | 3.92 | 0.80 |
| step 2441 (end) | 3.12 | 0.67 |

About two thirds of the drop comes in the first 500 steps. The run takes about 7.1 GPU hours at roughly 790 tokens per second, and peak memory stays near 10.6 GB. The recording ends with a prompt about Ulaanbaatar. The base model drifts into Russian, the Cyrillic language it saw far more of, and the control stays in Mongolian.

Evaluate step 0 with the same script before training. If the base number looks wrong, the eval is wrong, and you've lost nothing yet.

## Forgetting {#forgetting}
> Training only on Mongolian makes the model a little worse at English. About 5% is normal. A big jump means the training is too aggressive.

The lesson also tracks English perplexity on a WikiText sample: 6.10 before, 6.41 after, about 5% worse. The base weights are frozen, but the adapters change every layer's output, and they only ever saw Mongolian. If English matters for your use, mix a few percent of English into the training data. If the jump is much bigger, lower the learning rate or the rank.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Continued pretraining**: more next-token training on new text, starting from a trained model.
- **Control run**: a run that changes nothing except what's being tested, so other runs can be compared with it.
- **Held-out set**: documents never trained on, used only for evaluation.
- **BOS**: the token at the start of every sequence. It's never predicted.
- **Logits**: the model's raw scores, one row per position and one column per vocabulary entry.
- **NLL (negative log-likelihood)**: −ln of the probability the model gave the right token, in nats.
- **nats / bits**: units of the loss with the natural log or log base 2. One bit is ln 2 ≈ 0.693 nats.
- **Perplexity (per token)**: e to the average loss per predicted token. Comparable only between runs that share a tokenizer.
- **Pooled**: totals added across all documents before dividing, so every token counts once.
- **Bits per character**: total loss in bits divided by the number of characters. It doesn't depend on the tokenizer.
- **Bits per byte**: the same per UTF-8 byte. Cyrillic letters are 2 bytes, so it's about half of bits per character.
- **Forgetting**: getting worse at what the model used to know, here English, while learning something new.
- **GPU hours**: the run's wall-clock time on one GPU, the lesson's systems metric.
- **Mongolian held-out perplexity**: the lesson's quality metric, per token with Llama 3.1's tokenizer.
