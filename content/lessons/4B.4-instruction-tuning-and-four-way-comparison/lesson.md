---
id: "4B.4"
title: Instruction tuning and four-way comparison
summary: Instruction-tune the extended model with 4A.5's recipe, run 4A.5's eval on all four models, and score them fairly, both on quality and on what each answer costs.
phase: "4B"
hours: 10
type: research
status: ready
protected: false
cut: null
prerequisites:
  - 4B.3
  - 4A.5
papers: []
quality_metric:
  id: eval_score
  label: Best Mongolian eval score of the four models
  unit: "%"
  note: 4A.5's score, the mean of mn_qa and mn_mc. In the reference run the best model is extended + instruction tuning, and control + 4A.5's tuning scores 61.4%. Beat it.
  inputs:
    - { id: eval_score, label: Best score (%), hint: The top line of your comparison table }
  compute: value
  reference: 64.6
  pass: { min: 61.4 }
systems_metric:
  id: tokens_per_answer
  label: Mean tokens per Mongolian answer
  unit: tokens
  note: For the best model, over every answer in the eval, counted with its own tokenizer. The control, with Llama's tokenizer, averaged 11.8 in the reference run.
  inputs:
    - { id: tokens_per_answer, label: "Mean tokens per answer, best model", hint: Total answer tokens / answers }
  compute: value
  reference: 5.7
  pass: { max: 9.0 }
quiz:
  - id: why_macro
    q: 4A.5's eval has 150 mn_qa questions and 200 mn_mc questions. Why is the score the mean of the two task scores, rather than the share right over all 350?
    answer: 1
    options:
      - text: Pooling takes longer to compute
        why: Both take a fraction of a second. The difference is in what the number means.
      - text: Pooled, mn_mc is 57% of the score, so its easier four-option format would outweigh the short answers
        why: Right. Pooling weights each task by how many items it happens to have. The mean of task scores gives each skill an equal say, and adding items to one task doesn't shift the balance.
      - text: Pooling double-counts items that appear in both tasks
        why: Each item belongs to one task. The problem is the weights, not duplicates.
  - id: is_it_real
    q: On mn_qa's 150 items, extended + IT beats control by 4 points, 52.7% against 48.7%. Only 14 items differ, 10 won and 4 lost. What can you conclude?
    answer: 2
    options:
      - text: It's clearly better. 4 points is a big gap
        why: The size of the gap means little without the sample size. Only the 14 disagreements carry information, and 10 against 4 can happen by chance.
      - text: Nothing at all. 150 items is too few for any comparison
        why: 150 items can show a real difference if the gap is large. Here it's borderline, which is worth saying, not ignoring.
      - text: "It's suggestive, not settled: 10 of 14 is about 1.6 standard deviations from a coin flip"
        why: Right. If both models were equal, wins would be about 7 ± 1.9 out of 14, and 10 is 1.6σ out. Report it as suggestive, add items, or check whether the other task points the same way.
  - id: bigger_head
    q: The extended model's LM head has 136,256 rows instead of 128,256. Does that eat the savings from shorter answers?
    answer: 0
    options:
      - text: No. It adds about 0.4% compute per token, and answers are about 52% shorter
        why: "Right. 8,000 × 4,096 = 33M extra head weights on an 8B model, about 0.4% more FLOPs per token. Halving the tokens per answer halves the decode steps."
      - text: Yes. A 6% bigger vocab makes every token 6% slower
        why: Only the LM head grows by 6%, and it's a small part of the model. The 32 transformer layers don't change.
      - text: It depends on the GPU. On some cards it's slower overall
        why: Decoding cost is dominated by reading the weights, and the model grew by 0.8% (head and embedding). With half the tokens per answer it's faster on any card.
---

## Goal

Four models take 4A.5's Mongolian eval: **base** (Llama 3.1 8B as released), **control** (4A.4's continued pretraining with Llama's own tokenizer, then 4A.5's instruction tuning), **extended** (4B.3, new vocab, no instruction tuning) and **extended + IT** (4B.3 plus 4A.5's instruction tuning, which you run in this lesson). Each pair isolates one change: base → control is continued pretraining plus tuning, control → extended + IT is the tokenizer, extended → extended + IT is instruction tuning. Score them the way 4A.5 does, each task counting equally, and add what each answer costs in tokens. That table is the result of phase 4B.

## Learn

```card
type: concept
id: four_models
eyebrow: The comparison
title: Four models, one eval, and each pair answers one question.
visual:
  kind: bars
  max: 70
  bars:
    - { label: base, value: 30.5, text: 30.5% }
    - { label: control, value: 61.4, text: 61.4% }
    - { label: extended, value: 39.2, text: 39.2% }
    - { label: extended + IT, value: 64.6, text: 64.6% }
body: |
  The bars are the reference run's 4A.5 scores, the mean of mn_qa (short answers, exact match) and mn_mc (four options, scored by log-probability per character). Extended + IT beats control, so the new tokenizer helps on top of the same training and tuning. Extended without tuning trails both tuned models, but it beats the control before tuning (36.9%), the fair pair for it. Every model runs through the same harness with the same prompts, or the pairs stop meaning anything.
```

```card
type: calc
id: macro_calc
eyebrow: Do the math
title: What's the score, the mean of the two task scores?
given:
  - { label: mn_qa, value: 3 of 4 right }
  - { label: mn_mc, value: 5 of 8 right }
unit: "%"
answer: { min: 68.5, max: 69.0 }
placeholder: "0.0"
why_ok: (75 + 62.5) / 2 = 68.75%. Pooled it would be 8 of 12 = 66.7%, because mn_mc's 8 items outweigh mn_qa's 4.
why_wrong: Score each task first, then take the mean of those, so each task counts once however many items it has.
hint: Work out two percentages first, one per task, then average them.
```

```card
type: fill_blank
id: macro_blanks
eyebrow: Fill the blanks
title: Accuracy per task, then the mean over tasks.
code: |
  acc = {key: sum(r.correct for r in g) / [[0]] for key, g in by(rows, "model", "task").items()}
  mine = [a for (m, t), a in acc.items() if m == model]
  score = 100 * sum(mine) / [[1]]
tokens: [len(g), len(rows), len(mine), "2"]
answer: [len(g), len(mine)]
tips:
  len(rows): That's every answer from every model and task, so each accuracy comes out tiny. Divide by the items in this one (model, task) group.
  "2": That's right for this eval only. Add a third task, as 4B's own extensions might, and every score is off. Count the model's own tasks.
why_ok: Each (model, task) group's right answers over its own size is that task's accuracy. A model's score is the mean of its task accuracies, so every task counts once.
why_wrong: "Each division should use the size of the thing it's averaging: the items in one group, then the tasks of one model."
hint: The first division turns a count of right answers into an accuracy. The second turns a sum of accuracies into a mean.
```

```card
type: predict
id: paired_wins
eyebrow: Predict the output
title: Two models on the same 8 items (1 = right). What prints?
code: |
  a = [1, 1, 0, 1, 0, 1, 1, 0]
  b = [1, 0, 1, 1, 1, 1, 1, 1]
  only_a = sum(x > y for x, y in zip(a, b))
  only_b = sum(y > x for x, y in zip(a, b))
  print(only_a, only_b)
options:
  - text: "5 7"
    why: Those are the totals right, 5 for a and 7 for b. The paired count only looks at items where the two disagree.
  - text: "1 3"
    correct: true
    why: "Only a got item 2 right. Only b got items 3, 5 and 8. The 4 items both got right say nothing about which model is better, so a paired comparison leaves them out."
  - text: "4 4"
    why: 4 is the number of items both got right. Those agreements are left out on purpose.
  - text: "2 2"
    why: Count again where exactly one of the two is 1. b alone wins three items.
hint: Look only at the positions where a and b differ.
```

```card
type: calc
id: decode_saved
eyebrow: Do the math
title: What share of decode steps does extended + IT save against control?
given:
  - { label: Control, value: 11.8 tokens per answer }
  - { label: Extended + IT, value: 5.7 tokens per answer }
unit: "%"
answer: { min: 51, max: 52.5 }
placeholder: "0"
why_ok: 1 − 5.7 / 11.8 ≈ 51.7% fewer tokens, so about 52% fewer decode steps for the same answers. At a fixed time per token, answers come back in under half the time, with under half the KV cache.
why_wrong: The saving is the share of control's tokens that extended + IT doesn't need.
hint: Divide the extended model's tokens by the control's, then subtract that from 1.
```

```card
type: quiz
ref: why_macro
```

```card
type: quiz
ref: is_it_real
```

```card
type: quiz
ref: bigger_head
```

## Build steps

### Score the four models

```challenge
id: macro_scores
file: compare.py
minutes: 25
start:
  code: |
    from collections import namedtuple

    # A slice of your 4A.5 eval: the same 12 items for all four models. One row per answer: the model,
    # the task, the item, whether it was right (1 or 0), and the answer's length in that model's tokens.
    #   mn_qa  short answer, exact match   "Монгол Улсын нийслэл хаана байдаг вэ?" → "Улаанбаатар"
    #   mn_mc  four options                "Би сургууль ___ явна."  A) руу  B) рүү  C) луу  D) лүү → A
    Row = namedtuple("Row", "model task item correct tokens")
    ROWS = [Row(*r) for r in [
        ("base", "mn_qa", 1, 0, 12), ("base", "mn_qa", 2, 0, 9), ("base", "mn_qa", 3, 0, 14), ("base", "mn_qa", 4, 0, 11),
        ("base", "mn_mc", 1, 1, 8), ("base", "mn_mc", 2, 0, 11), ("base", "mn_mc", 3, 1, 7), ("base", "mn_mc", 4, 0, 10),
        ("base", "mn_mc", 5, 1, 9), ("base", "mn_mc", 6, 0, 12), ("base", "mn_mc", 7, 1, 6), ("base", "mn_mc", 8, 0, 9),
        ("control", "mn_qa", 1, 1, 10), ("control", "mn_qa", 2, 0, 9), ("control", "mn_qa", 3, 1, 13), ("control", "mn_qa", 4, 0, 8),
        ("control", "mn_mc", 1, 1, 8), ("control", "mn_mc", 2, 1, 10), ("control", "mn_mc", 3, 1, 7), ("control", "mn_mc", 4, 0, 11),
        ("control", "mn_mc", 5, 1, 9), ("control", "mn_mc", 6, 1, 12), ("control", "mn_mc", 7, 0, 6), ("control", "mn_mc", 8, 1, 9),
        ("extended", "mn_qa", 1, 0, 6), ("extended", "mn_qa", 2, 1, 5), ("extended", "mn_qa", 3, 0, 7), ("extended", "mn_qa", 4, 0, 5),
        ("extended", "mn_mc", 1, 1, 4), ("extended", "mn_mc", 2, 0, 6), ("extended", "mn_mc", 3, 1, 4), ("extended", "mn_mc", 4, 1, 5),
        ("extended", "mn_mc", 5, 0, 5), ("extended", "mn_mc", 6, 1, 6), ("extended", "mn_mc", 7, 0, 3), ("extended", "mn_mc", 8, 1, 5),
        ("extended_it", "mn_qa", 1, 1, 5), ("extended_it", "mn_qa", 2, 1, 5), ("extended_it", "mn_qa", 3, 1, 6), ("extended_it", "mn_qa", 4, 0, 4),
        ("extended_it", "mn_mc", 1, 1, 4), ("extended_it", "mn_mc", 2, 1, 5), ("extended_it", "mn_mc", 3, 0, 4), ("extended_it", "mn_mc", 4, 1, 6),
        ("extended_it", "mn_mc", 5, 1, 5), ("extended_it", "mn_mc", 6, 0, 6), ("extended_it", "mn_mc", 7, 0, 3), ("extended_it", "mn_mc", 8, 1, 5),
    ]]


    def by(rows, *fields):
        """Group rows by one field or several: by(ROWS, "model") → {"base": [rows], …},
        by(ROWS, "model", "task") → {("base", "mn_qa"): [rows], …}."""
        groups = {}
        for r in rows:
            key = getattr(r, fields[0]) if len(fields) == 1 else tuple(getattr(r, f) for f in fields)
            groups.setdefault(key, []).append(r)
        return groups

setup:
  - say: "These are eval results: 48 rows, one per answer, four models times 12 items. The tasks aren't the same size. mn_qa has 4 items here, mn_mc has 8."
    pause: 4.2
  - say: "by() groups rows for you. By model and task, it gives one list of rows for each pair."
    pause: 3.4
  - say: "Here's the stub. Score each model the way 4A.5 does, the mean of its task accuracies, so every task counts once. Your turn."
    type_file: macro_scores/stub.py
    pause: 1.2
tests: macro_scores/tests.py
solution: macro_scores/solution.py
alternatives:
  - macro_scores/alternatives/statistics_mean.py
  - macro_scores/alternatives/explicit_loop.py
hints:
  nudge: Two averages, one inside the other. First each (model, task) group's share of right answers, then, for each model, the mean of those shares.
  concept: "A group's accuracy is `sum(r.correct for r in group) / len(group)`, a number from 0 to 1. `acc` maps `(model, task)` keys to those. For one model, keep the values whose key starts with that model, average them, and multiply by 100. Every task counts once, whatever its size."
  skeleton: |
    acc = {key: sum(r.correct for r in group) / ___(_____) for key, group in by(rows, "model", "task").items()}
    mine = [a for (m, t), a in acc.items() if _ == _____]
    scores[model] = ___ * sum(____) / len(____)
  solution: |
    acc = {key: sum(r.correct for r in group) / len(group) for key, group in by(rows, "model", "task").items()}
    mine = [a for (m, t), a in acc.items() if m == model]
    scores[model] = 100 * sum(mine) / len(mine)
on_solve: "Look at control and extended_it. Pooled over all 12 items they tie, 8 right each. By task, extended_it leads 68.8% to 62.5%, because its extra right answer is on mn_qa, the smaller task, where each item counts for more. Neither number is wrong, but they answer different questions, and 4A.5's score is the per-task one. Real tasks have 150 and 200 items; 12 just make the effect easy to see."
mistakes:
  - { file: macro_scores/mistakes/placeholders.py, fails: one_score_per_model, says: "still `...`" }
  - { file: macro_scores/mistakes/fraction.py, fails: one_score_per_model, says: "between 0 and 1" }
  - { file: macro_scores/mistakes/pooled.py, fails: tasks_count_equally, says: "pooled" }
  - { file: macro_scores/mistakes/all_models.py, fails: tasks_count_equally, says: "Every model gets the same score" }
  - { file: macro_scores/mistakes/count_not_rate.py, fails: tasks_count_equally, says: "counts right answers" }
  - { file: macro_scores/mistakes/len_rows.py, fails: tasks_count_equally, says: "every row" }
  - { file: macro_scores/mistakes/two_tasks.py, fails: tasks_count_equally, says: "divide by 2 tasks" }
```

### Add what each answer costs

```challenge
id: answer_cost
file: compare.py
minutes: 20
start: { from: macro_scores }
setup:
  - say: "A score alone hides half the story. The extended models write the same answers in about half the tokens, and every generated token is a decode step you pay for."
    pause: 4.0
  - say: "Here's the stub: tokens per answer, and tokens per right answer, which charges each model for its wrong answers too. Your turn."
    type_file: answer_cost/stub.py
    pause: 1.2
tests: answer_cost/tests.py
solution: answer_cost/solution.py
alternatives:
  - answer_cost/alternatives/loops.py
hints:
  nudge: Both are a sum of tokens over a count. What do you divide by in each, every answer or only the right ones?
  concept: "`sum(r.tokens for r in group)` is every answer token the model produced in the eval. Divided by `len(group)` it's the mean answer length. Divided by `sum(r.correct for r in group)`, the number it got right, it's what one right answer costs, wrong answers included."
  skeleton: |
    return {model: sum(r.______ for r in group) / ___(_____) for model, group in by(rows, "model").items()}
    return {model: sum(r.______ for r in group) / sum(r._______ for r in group) for model, group in by(rows, "model").items()}
  solution: |
    return {model: sum(r.tokens for r in group) / len(group) for model, group in by(rows, "model").items()}
    return {model: sum(r.tokens for r in group) / sum(r.correct for r in group) for model, group in by(rows, "model").items()}
on_solve: "Extended + IT answers in about half of control's tokens, and pays about half per right answer too, since it gets at least as many right. Base is the worst on both: it gets few right, and spends Llama's 3.4 tokens a word on every answer. Score, tokens per answer and cost against control side by side: that table is the result of phase 4B."
mistakes:
  - { file: answer_cost/mistakes/placeholders.py, fails: tokens_per_answer, says: "still returns `...`" }
  - { file: answer_cost/mistakes/tpa_sum.py, fails: tokens_per_answer, says: "total" }
  - { file: answer_cost/mistakes/tpa_right_only.py, fails: tokens_per_answer, says: "only the right answers" }
  - { file: answer_cost/mistakes/tpr_mean_of_right.py, fails: tokens_per_right_answer, says: "wrong answers cost nothing" }
  - { file: answer_cost/mistakes/tpr_all.py, fails: tokens_per_right_answer, says: "every answer" }
  - { file: answer_cost/mistakes/tpr_inverted.py, fails: tokens_per_right_answer, says: "upside down" }
```

### Instruction-tune and evaluate

```replay
id: four_way_eval
intro: "On the GPU: instruction-tune the 4B.3 model with 4A.5's recipe (the same 9,777 pairs and 1,222 steps, continuing its r=32 adapter), then run 4A.5's harness on all four models, about 18 minutes each. Watch a recording first."
log: replays/four-way-eval.jsonl
speed: 40
recorded:
  hardware: RTX 3060 12 GB
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: Mean tokens per answer
  flag_above: 9
  x_labels: [base, extended + IT]
  caption: Red bars cost more than 9 tokens per answer. The two models with Llama's own tokenizer spend nearly twice as many tokens on the same answers.
summary:
  - { label: Best score, value: 64.6% (extended + IT) }
  - { label: vs control, value: +3.2 points (suggestive) }
  - { label: Tokens per answer, value: 11.8 → 5.7 }
run_yourself: |
  # on your GPU, from the course repo
  python sft.py --base runs/llama31-mn-8k-cpt/best --adapter runs/llama31-mn-8k-cpt \
    --data data/mn-instruct.jsonl --epochs 2 --mask-prompt --out runs/llama31-mn-8k-it
  python eval_mn.py --models base,runs/sft,runs/llama31-mn-8k-cpt,runs/llama31-mn-8k-it --tasks mn_qa,mn_mc
```

### Record your metrics

```metrics
id: record
intro: The best model's score and its mean tokens per answer. They're compared with the reference run and the pass limit.
```

## Hints

- Instruction-tune extended + IT with exactly 4A.5's data (the same 9,777 pairs, left after the 23 eval-overlapping examples were removed), LoRA rank, learning rate and steps. If anything differs, the control → extended + IT gap isn't only the tokenizer any more.
- Use 4A.5's harness unchanged: the same prompts, normalisation for mn_qa's exact match, and log-probability per character for mn_mc. Per character, not per token, so the extended tokenizer's longer tokens don't tilt mn_mc.
- Count answer tokens with each model's own tokenizer: that's what it actually decodes. For a tokenizer-free length, also report characters per answer.
- Run the paired comparison on control vs extended + IT for each task: count the items only one of them gets right. With n disagreements, a win count more than about √n away from n / 2 is worth trusting.
- If you want a task 4A.5 doesn't have, such as translation, add it as a 4B extension of the harness, run it on all four models, and report it separately from 4A.5's score.
- Write up the four-way table with the score, both task scores, tokens per answer, tokens per right answer and the English check side by side. It's the deliverable of phase 4B.

## Done when

- [ ] Per-task scores for all four models {#scores_done auto=challenge:macro_scores}
- [ ] Tokens per answer and per right answer {#cost_done auto=challenge:answer_cost}
- [ ] Four-way table from your own eval run, written up with the paired comparison {#table_written}
- [ ] Best score and its tokens per answer recorded {#metrics_recorded auto=metrics}
