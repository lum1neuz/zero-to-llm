---
id: "4A.5"
title: Instruction tuning and eval harness
summary: Teach the control model to follow Mongolian instructions with a loss that only scores the assistant's words, then build the eval harness that every 4B model is judged by.
phase: "4A"
hours: 4
type: research
status: ready
protected: false
cut: null
prerequisites:
  - 4A.4
papers:
  - title: "Training language models to follow instructions with human feedback"
    authors: Ouyang et al.
    year: 2022
    url: https://arxiv.org/abs/2203.02155
quality_metric:
  id: eval_score
  label: Mongolian eval score
  unit: "%"
  note: The mean of short-answer exact match and length-normalised multiple-choice accuracy, on your held-out Mongolian eval set. The control model before instruction tuning scores about 37%.
  inputs:
    - { id: eval_score, label: Eval score (%), hint: The score column of the harness's report }
  compute: value
  reference: 61.4
  pass: { min: 50.0 }
systems_metric:
  id: eval_minutes
  label: Full eval run time
  unit: min
  note: Every 4B model runs this harness at least once, so a fast harness pays for itself. Most of the time goes to generating short answers.
  inputs:
    - { id: eval_minutes, label: Eval run time (min), hint: The harness prints it at the end }
  compute: value
  reference: 18
  pass: { max: 30 }
quiz:
  - id: why_mask
    q: Why not let the loss score the user's and the system's tokens too?
    answer: 1
    options:
      - text: It would make training slower
        why: The forward and backward passes cost the same either way. Masking changes what the gradient pushes toward, not the speed.
      - text: The model would also learn to write prompts, and a long prompt would drown out the answer's signal
        why: Every scored token pulls the model toward producing it. Scoring the question teaches it to ask questions, and with long prompts most of the gradient comes from text the assistant should never write.
      - text: The user's tokens have no next token to predict
        why: Every position except the last has a next token. They're masked because predicting them isn't the skill you're training.
  - id: eot_trained
    q: Should the assistant's closing `<|eot_id|>` count in the loss?
    answer: 0
    options:
      - text: Yes, or the model never learns where its answer ends
        why: Generation stops when the model emits `<|eot_id|>`. If it's never a target, the model rambles on into a new turn of its own invention.
      - text: No, special tokens are always masked
        why: The header tokens are masked because the template adds them. The end-of-turn token is the one special token the model has to produce itself.
      - text: It doesn't matter, the tokenizer adds it at generation time
        why: At generation time nothing adds it. The model has to choose to stop, and it learns that from this one label.
  - id: contamination
    q: After instruction tuning, your eval score jumps from 37% to 91%. What do you check first?
    answer: 2
    options:
      - text: Nothing. Instruction tuning is known to help a lot
        why: It helps, as the reference run's 37% → 61% shows. A jump to 91% is more than this data can explain, and the first suspect is the data.
      - text: Whether the eval is too easy
        why: Possible, but it scored 37% before training on the same questions. Something about training changed, so look there first.
      - text: Whether eval questions leaked into the instruction data
        why: Translated instruction sets and web-scraped Q&A often contain the very questions your eval asks. Search the training data for long word overlaps with every eval item and remove the matches.
---

## Goal

Turn the control model from 4A.4 into an assistant. **Instruction tuning** trains on whole conversations, rendered with Llama 3.1's chat template, but scores only what the assistant says: every other position gets the label -100, which the loss skips. Then build the **eval harness**: exact match on short answers after normalizing them, and multiple choice by comparing log-probabilities fairly across options of different lengths. 4B.4 runs this harness on every model, so its numbers decide the comparison.

## Learn

```card
type: concept
id: template_and_mask
eyebrow: The idea
title: Train on the whole conversation, grade only the answer.
body: |
  A chat template wraps every turn in special tokens: `<|start_header_id|>user<|end_header_id|>` before the text, `<|eot_id|>` after it. The model reads all of it, so it learns from the context. But the loss only scores positions whose label is a real token id, and the labels are **-100** everywhere except the assistant's words and its closing `<|eot_id|>`. The model learns to answer, not to ask.
```

```card
type: predict
id: predict_labels
eyebrow: Predict the output
title: What does the label tensor hold?
code: |
  roles = [None, "user", "user", None, "assistant", "assistant"]
  ids = torch.tensor([0, 7, 8, 2, 9, 3])
  labels = ids.clone()
  labels[torch.tensor([r != "assistant" for r in roles])] = -100
  print(labels.tolist())
options:
  - text: "[-100, -100, -100, -100, 9, 3]"
    correct: true
    why: Every position whose role isn't "assistant" becomes -100, the template's None included. Only the assistant's two tokens keep their ids.
  - text: "[0, 7, 8, 2, -100, -100]"
    why: That's the mask the wrong way round, training on everything but the answer. `!=` picks the positions that aren't the assistant's.
  - text: "[-100, 7, 8, -100, 9, 3]"
    why: That masks only the template pieces. The user's tokens have the role "user", which isn't "assistant", so they're masked too.
hint: Which positions does the mask pick, and what gets written there?
```

```card
type: fill_blank
id: build_batch
eyebrow: Fill the blanks
title: Tokenize a chat and build its labels.
code: |
  text = tokenizer.apply_chat_template(messages, tokenize=False)
  ids = tokenizer(text, add_special_tokens=[[0]])["input_ids"]
  labels = [i if r == [[1]] else [[2]] for i, r in zip(ids, roles)]
tokens: ["False", "True", '"assistant"', '"user"', "-100", "0"]
answer: ["False", '"assistant"', "-100"]
tips:
  "True": "The template already starts with `<|begin_of_text|>`. Adding special tokens again gives two, a sequence the model never saw in pretraining."
  '"user"': Then the model learns to write the questions and never gets graded on its answers.
  "0": "0 is a real token id. A label of 0 trains the model to predict that token. The loss only skips -100."
why_ok: "The template supplies the special tokens, so the tokenizer mustn't add them again. Only the assistant's tokens keep their ids, and everything else is -100."
why_wrong: "Check each blank: who already added `<|begin_of_text|>`, whose words should be graded, and which label does the loss skip?"
hint: The loss skips one particular label. It isn't 0.
```

```card
type: predict
id: length_norm
eyebrow: Predict the output
title: Raw sums or per character. What prints?
code: |
  options = ["Төв", "Сүхбаатарын талбай"]
  logprobs = [-4.5, -10.8]   # summed over each option's tokens
  raw = max(range(2), key=lambda i: logprobs[i])
  norm = max(range(2), key=lambda i: logprobs[i] / len(options[i]))
  print(options[raw], "|", options[norm])
options:
  - text: "Төв | Төв"
    why: Per character, Төв scores −4.5 / 3 = −1.5 and Сүхбаатарын талбай −10.8 / 18 = −0.6. The long option wins once length is taken out.
  - text: "Төв | Сүхбаатарын талбай"
    correct: true
    why: Raw sums favour the 3-letter option just because it has fewer tokens to pay for. Per character, the 18-letter name is far more likely. The square is Sükhbaatar Square, so the normalised pick is right.
  - text: "Сүхбаатарын талбай | Сүхбаатарын талбай"
    why: The raw sums are −4.5 and −10.8, and −4.5 is the larger. Without normalising, the short option wins.
hint: "Compute both: the raw sums, then each divided by the option's length."
```

```card
type: reorder
id: sft_batch
eyebrow: Put it in order
title: Arrange one training example, from messages to loss.
lines:
  - text = tokenizer.apply_chat_template(messages, tokenize=False)
  - ids = tokenizer(text, add_special_tokens=False)["input_ids"]
  - labels = [i if r == "assistant" else -100 for i, r in zip(ids, roles)]
  - 'batch = {"input_ids": torch.tensor([ids]), "labels": torch.tensor([labels])}'
  - loss = model(**batch).loss
shuffle: [2, 4, 0, 3, 1]
why_ok: "Render, tokenize, label, batch, loss. A Hugging Face model shifts the labels inside `.loss`, so you pass labels lined up with the input ids. The `sft_loss` you write below shifts them itself."
why_wrong: "Follow the names. The labels need ids, and the batch needs both."
hint: The first line takes the messages. The last line needs a batch.
```

```card
type: quiz
ref: why_mask
```

```card
type: quiz
ref: eot_trained
```

```card
type: quiz
ref: contamination
```

## Build steps

### Mask the loss to the assistant's tokens

```challenge
id: loss_mask
file: sft.py
minutes: 30
start: { code: "import torch\nimport torch.nn.functional as F\n\n" }
setup:
  - say: "Here's a short conversation, rendered with Llama 3.1's chat template. To keep it readable each word is one toy token, and each piece remembers who said it. The template's own pieces have no speaker."
    type_file: loss_mask/chat.py
    pause: 3.4
  - say: "Instruction tuning feeds the model the whole sequence but only grades the assistant's part. The label -100 tells the loss to skip a position."
    pause: 3.4
  - say: "Here's the stub. Write build_labels and sft_loss. The table it prints shows every token's label."
    type_file: loss_mask/stub.py
    pause: 1.2
tests: loss_mask/tests.py
solution: loss_mask/solution.py
alternatives:
  - loss_mask/alternatives/where.py
  - loss_mask/alternatives/comprehension.py
  - loss_mask/alternatives/default_ignore.py
  - loss_mask/alternatives/manual_mean.py
hints:
  nudge: The labels are a copy of the ids with every non-assistant position overwritten. The loss is the usual next-token cross-entropy, skipping those positions.
  concept: "`ids.clone()` makes a separate tensor, so writing into it leaves the model's input alone. A bool tensor built from `roles` picks the positions to overwrite, `labels[mask] = -100`. Template pieces have the role None and the system and user turns have their own roles, so mask every position whose role isn't \"assistant\". In the loss, position t predicts the label at t + 1, and `ignore_index=-100` averages over only the positions that count."
  skeleton: |
    labels = ids._____()
    labels[torch.tensor([r ___ "assistant" for r in roles])] = ____
    return F.cross_entropy(logits[____], labels[___], ignore_index=____)
  solution: |
    labels = ids.clone()
    labels[torch.tensor([r != "assistant" for r in roles])] = -100
    return F.cross_entropy(logits[:-1], labels[1:], ignore_index=-100)
on_solve: "7 of the 29 predictions count: the six words of the answer and the assistant's `<|eot_id|>`, which teaches the model where to stop. The system prompt, the question and the template are context the model reads but isn't graded on. -100 is also cross_entropy's default `ignore_index`, which is why Hugging Face uses it. Step 2 is unlocked."
mistakes:
  - { file: loss_mask/mistakes/placeholders.py, fails: labels_masked, says: "`labels` is still `...`" }
  - { file: loss_mask/mistakes/no_clone.py, fails: labels_masked, says: "`ids` changed too" }
  - { file: loss_mask/mistakes/inverted.py, fails: labels_masked, says: "Your mask is the wrong way round" }
  - { file: loss_mask/mistakes/zero_fill.py, fails: labels_masked, says: "0 is a real token id here" }
  - { file: loss_mask/mistakes/headers_kept.py, fails: labels_masked, says: "The template's own tokens keep their ids" }
  - { file: loss_mask/mistakes/user_only.py, fails: labels_masked, says: "The system message keeps its ids" }
  - { file: loss_mask/mistakes/loss_placeholder.py, fails: loss_value, says: "`sft_loss` still returns `...`" }
  - { file: loss_mask/mistakes/no_shift.py, fails: loss_value, says: "Position t predicts the *next* token" }
  - { file: loss_mask/mistakes/summed.py, fails: loss_value, says: "Use the mean" }
  - { file: loss_mask/mistakes/all_positions.py, fails: loss_value, says: "Average over the positions that count" }
```

### Score answers like the harness does

```challenge
id: eval_scoring
file: harness.py
minutes: 25
start: { code: "# A tiny Mongolian eval harness\n\n" }
setup:
  - say: "Four short-answer questions with the model's answers, and four multiple-choice questions with the log-probability the model gave each option, all in Mongolian."
    type_file: eval_scoring/data.py
    pause: 3.0
  - say: "Two ways to score. Exact match compares answers after normalizing them. Multiple choice picks the option the model finds most likely, which takes care when the options differ in length."
    pause: 3.6
  - say: "Here's the stub. Write normalize_answer and pick_option. The last lines print the harness's score."
    type_file: eval_scoring/stub.py
    pause: 1.2
tests: eval_scoring/tests.py
solution: eval_scoring/solution.py
alternatives:
  - eval_scoring/alternatives/regex.py
  - eval_scoring/alternatives/zip_max.py
hints:
  nudge: "`normalize_answer`: lowercase, filter the characters, then split and rejoin. `pick_option`: compare each option's log-probability divided by its length."
  concept: "`s.lower()` handles Cyrillic capitals too. Keep a character if `c.isalnum() or c.isspace()`, so letters and digits stay and punctuation goes. `\" \".join(text.split())` collapses runs of whitespace and trims the ends. An option's log-probability is a sum over its tokens, so a longer option scores lower just for being longer. Dividing by `len(option)` compares them per character. `max(range(n), key=...)` returns the index with the largest key."
  skeleton: |
    return " ".join("".join(c for c in s._____() if c._______() or c.isspace())._____())
    return ___(range(len(options)), key=lambda i: logprobs[i] / ___(options[i]))
  solution: |
    return " ".join("".join(c for c in s.lower() if c.isalnum() or c.isspace()).split())
    return max(range(len(options)), key=lambda i: logprobs[i] / len(options[i]))
on_solve: "Exact match 75%: \"Орхон гол\" against \"Орхон\" still misses, which is exact match being strict on purpose. Multiple choice gets all four. With raw sums it would get two, picking Богд and Төв because they're short. Report the length-normalised number, and say that you did."
mistakes:
  - { file: eval_scoring/mistakes/placeholders.py, fails: normalized, says: "`normalize_answer` still returns `...`" }
  - { file: eval_scoring/mistakes/no_lower.py, fails: normalized, says: "Lowercase first" }
  - { file: eval_scoring/mistakes/no_punct.py, fails: normalized, says: "the full stop makes it miss" }
  - { file: eval_scoring/mistakes/letters_only.py, fails: normalized, says: "Keep the digits" }
  - { file: eval_scoring/mistakes/no_spaces.py, fails: normalized, says: "every space is gone" }
  - { file: eval_scoring/mistakes/no_collapse.py, fails: normalized, says: "split the words and join them with single spaces" }
  - { file: eval_scoring/mistakes/raw_sum.py, fails: picks_option, says: "raw sums favour short options" }
  - { file: eval_scoring/mistakes/argmin.py, fails: picks_option, says: "You pick the *lowest* score" }
  - { file: eval_scoring/mistakes/returns_text.py, fails: picks_option, says: "returns the option's text" }
```

### Instruction-tune the control and run the harness

```replay
id: sft_and_eval
intro: "Instruction tuning continues the control run's adapter on about 10,000 Mongolian instruction pairs, with the loss masked to the assistant's turns. It takes under 2 hours. Then the harness scores three models on the same held-out questions: base Llama, the control, and the control after instruction tuning."
log: replays/sft-eval.jsonl
speed: 40
recorded:
  hardware: RTX 3060 12 GB
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
summary:
  - { label: Base Llama 3.1 8B, value: 30.5% }
  - { label: Control (4A.4), value: 36.9% }
  - { label: Control + instruction tuning, value: 61.4% }
  - { label: Eval run time, value: 18 min }
run_yourself: |
  # on your machine or Colab, from the course repo
  python sft.py --base meta-llama/Llama-3.1-8B --adapter runs/control \
    --data data/mn-instruct.jsonl --epochs 2 --mask-prompt --out runs/sft
  python eval_mn.py --models base,runs/control,runs/sft --tasks mn_qa,mn_mc
```

### Record your metrics

```metrics
id: record
intro: Your instruction-tuned model's eval score and how long the harness took. They're compared with the reference run and the pass limits.
```

## Hints

- Decontaminate before training: search the instruction data for any long run of words (13 or so) shared with an eval question, and drop those examples.
- Keep the eval questions out of everything: the corpus, the instruction data, your prompts while debugging. Write them once, store them apart, and only score against them.
- Report both multiple-choice numbers, raw and length-normalised. If they disagree a lot, the options differ in length, and that's worth knowing.
- Exact match undercounts right answers in other words ("Орхон гол" for "Орхон"). Read 20 misses by hand before trusting the number, and keep the gold answers short.
- Greedy decoding with a short limit (32 new tokens) keeps short-answer eval fast and repeatable. Sampling makes every run's score different.

## Done when

- [ ] Only the assistant's tokens, and its end-of-turn token, are scored by your loss {#loss_mask auto=challenge:loss_mask}
- [ ] Your harness normalizes answers and compares options per character {#eval_scoring auto=challenge:eval_scoring}
- [ ] Eval score and harness run time recorded for your instruction-tuned model {#metrics_recorded auto=metrics}
