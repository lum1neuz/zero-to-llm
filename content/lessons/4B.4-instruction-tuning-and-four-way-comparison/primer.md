---
gist: Four models take the same Mongolian eval, and each pair of them isolates one change. This lesson instruction-tunes the extended model, scores all four fairly, adds what each answer costs, and asks whether the differences are bigger than noise.
minutes: 16
---

## What this lesson is for {#why}
> Phase 4B ends with one table: four models, their scores, and what their answers cost. Built carefully, it says whether the Mongolian tokenizer was worth it.

You now have a model with a Mongolian-sized vocabulary that reads Mongolian well (4B.3). But it has only learned to continue text. An eval asks questions and expects answers, and answering is what **instruction tuning** teaches. So this lesson does two things:

1. Instruction-tune the extended model with exactly 4A.5's recipe.
2. Run 4A.5's eval on four models and build a fair comparison.

The browser challenges work on a small slice of eval results: scoring them the way 4A.5 does, then adding the cost of each answer in tokens.

## What you need from before {#before}
> 4A.5 built the eval harness and the instruction-tuning recipe. This lesson reuses both unchanged, so the only difference between the control and the extended model is the tokenizer.

From 4A.5:

- **Instruction tuning (SFT)**: supervised fine-tuning on question-and-answer pairs. 4A.5 used 9,777 Mongolian pairs, 1,222 steps, continuing the control's LoRA adapter. 23 pairs that overlapped eval questions were removed first.
- **mn_qa**: 150 short-answer questions. The model answers greedily, up to 32 new tokens, and is right if its answer matches exactly after **normalising**: lowercase, punctuation removed, spaces collapsed.
- **mn_mc**: 200 four-option questions. No answer is generated. The model scores each option by its log-probability per character, and picks the best.
- **The score**: the mean of the two task accuracies.

An example of each, with meanings:

| Task | Question | Answer |
|---|---|---|
| mn_qa | Монгол Улсын нийслэл хаана байдаг вэ? ("Where is Mongolia's capital?") | Улаанбаатар (Ulaanbaatar) |
| mn_mc | Би сургууль ___ явна. ("I go ___ school.") A) руу B) рүү C) луу D) лүү | A, руу ("to") |

The mn_mc options are four spellings of the same suffix. Mongolian picks between them by the vowels and last letter of the word before, so the question tests real grammar.

## Four models, one change per pair {#models}
> Every model takes the same eval through the same harness. Compare them in pairs, where only one thing differs.

| Model | What it is |
|---|---|
| base | Llama 3.1 8B as released |
| control | 4A.4's continued pretraining, then 4A.5's tuning |
| extended | 4B.3's model: new vocab, continued pretraining, no tuning |
| extended + IT | 4B.3's model plus 4A.5's tuning, from this lesson |

The pairs that mean something:

| Pair | What differs |
|---|---|
| control vs extended + IT | only the tokenizer |
| extended vs extended + IT | only the instruction tuning |
| extended vs control before tuning | only the tokenizer, before any tuning |

That's why every detail has to match: the same tuning data, steps and settings, and the same prompts, generation limits and scoring. If anything else differs, a gap no longer measures the tokenizer.

## Scoring: average the tasks, not the items {#macro}
> Score each task first, then average the task scores, so each task counts once however many items it has.

Pooling every item looks simpler, but it weights each task by its size. With 150 mn_qa and 200 mn_mc items, pooled scoring makes mn_mc 57% of the result.

The lesson's slice has 4 mn_qa items and 8 mn_mc items per model. Two of the models:

| Model | mn_qa | mn_mc | Pooled | Task average |
|---|---|---|---|---|
| control | 2 / 4 | 6 / 8 | 8 / 12 = 66.7% | (50 + 75) / 2 = 62.5% |
| extended + IT | 3 / 4 | 5 / 8 | 8 / 12 = 66.7% | (75 + 62.5) / 2 = 68.75% |

Pooled, they tie. Per task, extended + IT leads, because its extra right answer is on the smaller task, where each item weighs more. Neither number is wrong, but they answer different questions, and 4A.5's score is the per-task average. Use the same one for all four models.

Mistakes the tests check for: pooling all items, leaving scores as fractions instead of percentages, averaging every model's tasks together (all models then get the same score), counting right answers instead of the share right, dividing by every row in the eval, and hard-coding the number of tasks.

## Why mn_mc scores per character {#per-char}
> A multiple-choice option's score is its total log-probability divided by its length in characters. Per character keeps the comparison fair across options and across tokenizers.

The model doesn't write anything for mn_mc. It reads each option and adds up the log-probabilities of its tokens. Longer options collect more negative terms, so a raw total favours short options. Dividing by the number of tokens would fix that within one model, but the extended tokenizer cuts the same option into fewer tokens than Llama's does, so per-token scores would differ between the models for reasons that have nothing to do with Mongolian. Per character, the same option has the same length for every model.

It's the same idea as bits per character in 4A.4 and 4B.3: measure in units of text, not units of tokenizer.

## What each answer costs {#cost}
> Count every answer in the tokens of the model that produced it. Tokens per answer is the cost of a typical answer, and tokens per right answer charges the wrong answers too.

Each token is one decode step, one slot of KV cache, and one unit on any per-token bill. Two measures:

- **Tokens per answer**: all answer tokens divided by all answers.
- **Tokens per right answer**: all answer tokens divided by the answers that were right. A model that is often wrong pays for its wrong answers too, so this rises.

On the lesson's slice:

| Model | Tokens per answer | Tokens per right answer |
|---|---|---|
| base | 9.8 | 29.5 |
| control | 9.3 | 14.0 |
| extended | 5.1 | 10.2 |
| extended + IT | 4.8 | 7.2 |

Mistakes the tests check for: returning the total instead of the mean, averaging only the right answers (wrong answers cost tokens too), dividing tokens per right answer by every answer, and dividing upside down.

The bigger vocabulary has a cost too: the LM head has 8,000 more rows. That adds 8,000 × 4,096 ≈ 33M weights to an 8B model, about 0.4% more work per token. Answers about half as long more than pay for it.

## Is the difference real? {#paired}
> Only the items where two models disagree tell you which is better. Count the wins on each side and check whether the split is far from a coin flip.

When both models get an item right, or both get it wrong, it says nothing about which is better. So a **paired comparison** looks only at disagreements:

```
item:   1  2  3  4  5  6
A:      1  0  1  1  0  1
B:      1  1  1  0  1  1

only A right: item 4      → 1
only B right: items 2, 5  → 2
```

If the two models were equally good, each disagreement would be a coin flip. With n disagreements, the wins would land near n / 2, give or take about √n / 2. A split more than about √n away from n / 2 is worth trusting.

The reference run, control vs extended + IT:

| Task | Items | Disagree | Wins | Distance from a coin flip |
|---|---|---|---|---|
| mn_qa | 150 | 14 | 10 to 4 | 1.6σ |
| mn_mc | 200 | 31 | 18 to 13 | 0.9σ |

Neither task alone is decisive. Both point the same way, which makes the result suggestive. Report it like that, not as a proven win.

## Noise between runs {#noise}
> Training has randomness too. A gap smaller than what a different seed would produce isn't evidence of anything.

The eval itself is repeatable: greedy decoding and fixed questions give the same answers every time. Training isn't. A different random seed shuffles the data differently and starts the LoRA matrices differently, and the scores move a little. You won't know by how much from one run.

If you can afford it, repeat the instruction tuning with a second seed for control and extended + IT, and see how far each score moves. A difference smaller than that movement isn't a result. If you can't afford it, say so in the write-up and lean on the paired comparison.

## What you'll run {#run}
> On your GPU: about 1.5 hours of instruction tuning, then 4A.5's harness on all four models, about 18 minutes each. The recording in the lesson is illustrative.

The tuning continues 4B.3's LoRA adapter with 4A.5's 9,777 pairs and 1,222 steps. The new vocabulary rows stay frozen during tuning, as the control's embeddings did. Then the harness runs all four models.

Healthy output looks like this (reference run, illustrative):

| Model | mn_qa | mn_mc | Score | Tokens per answer |
|---|---|---|---|---|
| base | 14.0% | 47.0% | 30.5% | 12.4 |
| control | 48.7% | 74.0% | 61.4% | 11.8 |
| extended | 23.3% | 55.0% | 39.2% | 6.0 |
| extended + IT | 52.7% | 76.5% | 64.6% | 5.7 |

Extended without tuning trails both tuned models. It still beats the control before tuning, 36.9% in 4A.5, which is its fair pair. You record the best score (pass: at least 61.4%, the control's) and that model's tokens per answer (pass: at most 9). Then write up the table, with the paired comparison, as the result of phase 4B.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Instruction tuning (SFT)**: fine-tuning on question-and-answer pairs so the model answers instead of only continuing text.
- **Eval harness**: the fixed set of questions, prompts and scoring rules from 4A.5.
- **mn_qa**: 150 short-answer questions, scored by exact match after normalising.
- **mn_mc**: 200 four-option questions, scored by log-probability per character.
- **Exact match**: right only if the normalised answer equals the normalised reference.
- **Normalising**: lowercasing, removing punctuation and collapsing spaces before comparing.
- **Log-probability**: the log of the probability the model gives a piece of text. Higher (closer to 0) is more likely.
- **Pooled accuracy**: right answers over all items, so bigger tasks weigh more.
- **Macro average (task average)**: the mean of the per-task accuracies, so each task counts once.
- **Tokens per answer**: answer tokens divided by answers, in each model's own tokens.
- **Tokens per right answer**: answer tokens divided by right answers. It charges wrong answers too.
- **Paired comparison**: comparing two models on the same items, counting only the items where they disagree.
- **σ (sigma)**: a standard deviation, here how far a win count is from what a coin flip would give.
- **Seed**: the number that fixes a run's randomness. A different seed gives a slightly different model.
- **Four-way comparison**: the table of base, control, extended and extended + IT that ends phase 4B.
