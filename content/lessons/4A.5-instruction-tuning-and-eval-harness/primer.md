---
gist: Instruction tuning turns a model that continues text into one that answers, by training on conversations but grading only the assistant's words. An eval harness then scores every model the same way, so 4B's comparisons mean something.
minutes: 15
---

## What this lesson is for {#why}
> The control model writes Mongolian but doesn't answer questions. You'll teach it to, and build the test suite every later model is scored with.

After 4A.4, ask the control model "What is the capital of Mongolia?" and it may well continue with three more questions, because that's what text on the web often does next. It predicts text, it doesn't respond. **Instruction tuning** fixes that by training on conversations: a user asks, an assistant answers.

The second half of the lesson is the **eval harness**: a fixed set of held-out Mongolian questions, scored the same way for every model. Phase 4B runs it on each of its models, so its numbers decide the comparison. If you've worked on a web app, think of it as the regression test suite that every build has to pass before you compare builds.

## What you need from before {#before}
> Next-token loss with a shift by one, cross-entropy on logits, LoRA adapters, and the control model from 4A.4.

- **The shift** (4A.4). The logits at position t predict the token at t + 1.
- **`F.cross_entropy`** takes logits and target ids and returns the average loss. It also takes `ignore_index`: any target equal to that number is skipped. Its default is −100.
- **The control** (4A.4). Llama 3.1 8B plus a QLoRA adapter trained on 20M Mongolian tokens. Instruction tuning continues training that same adapter.

## From continuing text to answering {#sft}
> Supervised fine-tuning trains on example conversations, so the model learns what an assistant's reply looks like.

**Supervised fine-tuning (SFT)** is ordinary next-token training on a special kind of data: (instruction, answer) pairs, written or checked by people. The reference run uses 9,777 Mongolian pairs, 6,100 translated and checked by hand and 3,700 written from scratch, after 23 pairs that overlapped the eval were removed. Two passes over them take 1,222 steps and under 2 hours.

The training objective doesn't change. What changes is what the model sees and which tokens get graded.

## Chat templates {#template}
> A chat template wraps every turn in special tokens that mark who is speaking and where each turn ends.

The model sees one long sequence of tokens, so the conversation's structure has to be written into it. Llama 3.1's template does this with special tokens, which the tokenizer treats as single tokens:

| Piece | What it marks |
|---|---|
| `<\|begin_of_text\|>` | the start of the sequence (BOS) |
| `<\|start_header_id\|>` | a speaker's name follows |
| `user`, `assistant`, `system` | who's speaking |
| `<\|end_header_id\|>` | the name ends and the text begins |
| `<\|eot_id\|>` | end of this turn |

A user turn followed by an answer reads like this, one piece per line here so it fits a phone:

```
<|begin_of_text|>
<|start_header_id|>user
<|end_header_id|>
Монгол Улсын нийслэл
аль хот вэ?<|eot_id|>
<|start_header_id|>assistant
<|end_header_id|>
Улаанбаатар хот.<|eot_id|>
```

The question asks "Which city is the capital of Mongolia?" and the answer is "Ulaanbaatar city."

One classic bug: the template already adds `<|begin_of_text|>`. If you then tokenize the result with the tokenizer's default settings, it adds another one, and the model sees two BOS tokens in a row, which it never saw in pretraining. When you tokenize templated text, turn special tokens off.

## Loss masking {#masking}
> Every position whose label isn't the assistant's speech gets the label −100, and the loss skips it. The model reads the whole conversation but is graded only on the answer.

The labels start as a copy of the token ids. Then every position that isn't part of the assistant's reply is overwritten with −100:

| Token | Who | Label |
|---|---|---|
| `<\|begin_of_text\|>` | template | −100 |
| `user` (header) | template | −100 |
| question words | user | −100 |
| `<\|eot_id\|>` | user | −100 |
| `assistant` (header) | template | −100 |
| answer words | assistant | their ids |
| `<\|eot_id\|>` | assistant | its id |

Why each choice:

- **The question is masked**, or the model would also learn to write questions, and long prompts would drown out the answer's signal.
- **The template's tokens are masked**, because the template adds them. The model never has to write them.
- **The system prompt is masked**, or the model learns to produce system prompts.
- **The assistant's `<|eot_id|>` is kept.** Generation stops when the model emits it. If it's never a target, the model doesn't learn where its answer ends and rambles on.

Two classic mistakes:

- **Filling with 0 instead of −100.** 0 is a real token id. In the browser task it's `<|begin_of_text|>`, so the loss would train the model to predict it after every prompt word.
- **Forgetting to copy.** `labels = ids` in Python doesn't copy the tensor. It gives the same object a second name, exactly like assigning one Java array variable to another. Writing −100 into `labels` would also overwrite the model's input. Make a real copy first.

## Averaging over the right tokens {#mean}
> The loss is the mean over the graded positions only. Dividing by every position would make a long prompt shrink the loss without the answer getting any better.

In the browser task, the conversation has 30 tokens, so 29 positions predict something, but only 7 of them are graded: six answer words and the closing `<|eot_id|>`. The loss must be the average over those 7. `cross_entropy` with `ignore_index=−100` does exactly that.

The tests check three ways this goes wrong: no shift (scoring a position against the token it was just given), a sum instead of a mean (so a long answer counts as a bigger step than a short one), and a sum divided by all 29 positions.

Hugging Face models shift the labels themselves inside `model(...).loss`, so with them you pass labels lined up with the input ids. The `sft_loss` you write in the browser does the shift itself.

## What an eval harness is {#harness}
> A fixed set of held-out questions, scored the same way for every model, run as one command.

To compare models you need the same questions, the same scoring and the same settings every time. The harness in this course has two tasks:

| Task | Questions | Scored by |
|---|---|---|
| `mn_qa` | 150 short-answer | exact match after normalizing |
| `mn_mc` | 200 four-option | log-probability per character |

The score is the average of the two. Generation is **greedy** (always the most likely next token, up to 32 new tokens), so the same model gives the same answers every run. A full run takes about 18 minutes.

## Exact match and normalizing {#exact}
> Normalize both answers (lowercase, drop punctuation, collapse spaces) before comparing, but keep the words and digits themselves strict.

The model says "Хоёр." and the reference answer is "хоёр" ("two"). They mean the same, but `==` says no. Normalizing both sides first fixes it:

| Model's answer | Reference | Equal after normalizing? |
|---|---|---|
| `улаанбаатар.` | `Улаанбаатар` | yes: case, full stop |
| `Хоёр` | `хоёр` | yes: case |
| ` 1162 ` | `1162` | yes: spaces |
| `Орхон гол` | `Орхон` | no: an extra word |

The last row is exact match being strict on purpose. "Орхон гол" means "the Orkhon river", which is right, but the scorer can't know that. Read some misses by hand before trusting the number.

The tests catch normalizers that go too far as well: dropping digits makes every year match every other year, and removing all spaces runs words together.

## Multiple choice by log-probability {#mc}
> Score each option by how likely the model finds it, divided by its length. Raw sums favour short options just for being short.

For a multiple-choice question, the harness asks the model how likely each option's text is as the continuation, by adding up the log-probabilities of its tokens. Each extra token costs log-probability, so longer options score lower even when they're right:

| Option | Summed log-prob | Characters | Per character |
|---|---|---|---|
| Төв | −4.5 | 3 | −1.5 |
| Сүхбаатарын талбай | −10.8 | 18 | −0.6 |

The question asks for the name of Ulaanbaatar's central square. Raw sums pick the short "Төв" ("centre"). Per character, the right answer, Sükhbaatar Square, wins clearly. Log-probabilities are negative, so the best option is the one *closest to zero*, the maximum. Picking the minimum is a classic slip. Report the normalized number and say that you did.

## Contamination, and what you'll run {#run}
> Check that no eval question leaked into the training data, then train, then run the harness on three models.

If eval questions appear in the training data, the model is graded on answers it memorised. Translated instruction sets and scraped Q&A often contain common quiz questions. Before training, the reference run searched the instruction data for any run of 13 words shared with an eval question, and removed the 23 pairs it found. A sudden jump to a suspiciously high score is the usual sign that this step was skipped.

On your GPU you continue the control's adapter on the instruction pairs, then score three models. The replay is an **illustrative** recording written for the course, not a real run. Healthy looks like this:

| Model | mn_qa | mn_mc | Score |
|---|---|---|---|
| base Llama 3.1 8B | 14.0% | 47.0% | 30.5% |
| control (4A.4) | 21.3% | 52.5% | 36.9% |
| control + instruction tuning | 48.7% | 74.0% | 61.4% |

Training loss falls from about 1.4 to about 1.0, and validation loss ends near 1.08.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Instruction tuning / SFT (supervised fine-tuning)**: training on example conversations so the model learns to answer.
- **Instruction pair**: one user request and the answer the assistant should give.
- **Chat template**: the format that wraps each turn in special tokens.
- **Special tokens**: `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`, each a single token.
- **`add_special_tokens=False`**: tells the tokenizer not to add its own BOS, because the template already did.
- **Labels**: the target token at each position. −100 means "skip this one".
- **Loss masking**: setting the labels of everything except the assistant's reply to −100.
- **`ignore_index`**: the label value `cross_entropy` skips. Its default is −100.
- **`clone()`**: makes a real copy of a tensor, not a second name for the same one.
- **Eval harness**: a fixed set of held-out questions and scoring rules, run the same way on every model.
- **Exact match (EM)**: the share of answers equal to the reference after normalizing.
- **Normalize**: lowercase, drop punctuation and collapse whitespace before comparing answers.
- **Log-probability**: the natural log of a probability. Always negative or zero, and closer to zero is more likely.
- **Length-normalised accuracy**: multiple-choice accuracy with each option's log-probability divided by its length.
- **Greedy decoding**: always choosing the most likely next token, so runs are repeatable.
- **Contamination**: eval questions appearing in the training data.
- **Decontamination**: removing training examples that overlap the eval, here by shared 13-word runs.
- **Eval score**: the lesson's quality metric, the average of mn_qa and mn_mc.
- **Eval run time**: the lesson's systems metric, how long one harness run takes.
