---
id: "4B.3"
title: Continued pretraining
summary: Train the extended Llama 3.1 8B on exactly the control run's Mongolian text, plan its steps, hours and learning-rate schedule, and beat the control in bits per character.
phase: "4B"
hours: 20
type: run
status: ready
protected: false
cut: null
prerequisites:
  - 4B.2
papers:
  - title: Don't Stop Pretraining
    authors: Gururangan et al.
    year: 2020
    url: https://arxiv.org/abs/2004.10964
  - title: Simple and Scalable Strategies to Continually Pre-train Large Language Models
    authors: Ibrahim et al.
    year: 2024
    url: https://arxiv.org/abs/2403.08763
quality_metric:
  id: mn_ppl
  label: Mongolian held-out loss vs the control run
  unit: "%"
  note: Bits per character on 4A.3's held-out split, against your 4A.4 control run. Per-token perplexity can't be compared across tokenizers; bits per character can. Negative means the extended model is better.
  inputs:
    - { id: bpc_control, label: Control run bits per character, hint: From your 4A.4 eval log }
    - { id: bpc_extended, label: Extended run bits per character, hint: "Same held-out split, best checkpoint" }
  compute: pct_change
  reference: -4.2
  pass: { max: 0 }
systems_metric:
  id: gpu_hours
  label: GPU hours
  unit: h
  note: Training plus evals. The same text in fewer tokens should cost less than the control's 7.1 hours.
  inputs:
    - { id: gpu_hours, label: GPU hours for the run, hint: From your training log }
  compute: value
  reference: 3.6
  pass: { max: 7.1 }
quiz:
  - id: same_text
    q: Why train on exactly the control's 20M-token sample, rather than on 20M extended-vocab tokens?
    answer: 1
    options:
      - text: Because 20M is the most tokens a 12 GB GPU can train on
        why: The token count has nothing to do with memory. Memory depends on the batch and the model, not on how long you train.
      - text: 20M extended tokens is nearly twice the text, so a win could come from more data instead of the tokenizer
        why: Right. At 1.62 instead of 3.38 tokens per word, 20M extended tokens hold about 2.1× the documents. Same documents, same epoch, and the only thing that differs is the tokenizer.
      - text: The new rows need exactly 20M tokens to train
        why: Nothing magic about 20M. It's simply what the control used, and the comparison only means something if both runs see the same text.
  - id: overfit_signal
    q: Late in a longer run, train loss keeps falling but held-out bits per character start rising. What's happening?
    answer: 0
    options:
      - text: The model has started memorising the training text
        why: Right. After a few passes over a small sample, the model learns the training documents themselves rather than Mongolian. Stop at the best held-out checkpoint, not the last one.
      - text: The learning rate is too low to make progress
        why: A low learning rate slows train and held-out loss down together. Here they've split apart, which is a different signal.
      - text: The held-out split is broken
        why: It could be, but a held-out split that worked for a thousand steps doesn't break by itself. Train and held-out drifting apart is the classic sign of overfitting.
  - id: why_warmup
    q: Why ramp the learning rate up over the first 50 steps instead of starting at its peak?
    answer: 2
    options:
      - text: The first batches are the most important, so they get the smallest steps
        why: The first batches aren't special. Warmup is about the optimizer's state, not the data.
      - text: To save GPU time at the start of the run
        why: Every step costs the same time whatever the learning rate is. Warmup doesn't save compute.
      - text: Adam's estimates of each weight's gradient size start out empty, so early steps at full size can wreck pretrained weights
        why: Right. Adam divides by a running average of squared gradients, which needs some steps to settle. The 8,000 fresh rows also get large gradients at first. Small early steps keep both from knocking Llama's weights off course.
---

## Goal

Your extended Llama 3.1 8B starts at 3.83 nats per token on Mongolian, about 1.08 bits per character, but its 8,000 new rows have never been trained. **Continued pretraining** keeps training on next-token loss, on Mongolian text, from Llama's own weights: the new rows in full, everything else through LoRA with the control's settings. The comparison with 4A.4 is only fair on **the same documents**, so you train on the control's 20M-token sample, one epoch. Plan it first: how many extended tokens that text is, how many steps and hours it takes, and the learning-rate schedule. Then run it and compare with the control in **bits per character**.

## Learn

```card
type: concept
id: what_trains
eyebrow: The run
title: Same text, same loss, same settings. Only the tokenizer differs.
visual:
  kind: bars
  max: 1.1
  bars:
    - { label: Llama 3.1 8B, value: 1.04, text: "1.04" }
    - { label: control 4A.4, value: 0.67, text: "0.67" }
    - { label: extended + CPT, value: 0.642, text: "0.642" }
body: |
  Continued pretraining (CPT) is next-token prediction on Mongolian, starting from Llama's weights. Against 4A.4's control run, exactly one thing changes: the tokenizer, with its 8,000 new rows trained in full. The documents, the epoch, the LoRA rank, the learning rate and its schedule stay the same. The bars are held-out **bits per character** in the reference run, the number that means the same thing whichever tokenizer cut the text.
```

```card
type: calc
id: same_text_calc
eyebrow: Do the math
title: The control trained on 20M Llama tokens. How many extended tokens is the same text?
given:
  - { label: Control sample, value: 20M Llama 3.1 tokens }
  - { label: Llama 3.1, value: 3.38 tokens per word (1.6) }
  - { label: Extended, value: 1.62 tokens per word }
unit: M tokens
answer: { min: 9.45, max: 9.75 }
placeholder: "0.0"
why_ok: 20M / 3.38 ≈ 5.92M words, × 1.62 ≈ 9.59M extended tokens. The same documents in under half the tokens, so under half the steps.
why_wrong: Turn the Llama tokens into words first, then the words into extended tokens.
hint: How many words are 20M Llama tokens? Then how many extended tokens do those words cost?
```

```card
type: fill_blank
id: cosine_progress
eyebrow: Fill the blanks
title: How far through the decay is this step?
code: |
  progress = (step - [[0]]) / (total - [[1]])
  lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
tokens: [warmup, "0", warmup, total, step]
answer: [warmup, warmup]
tips:
  "0": Then the decay starts at step 0, during warmup. At step `warmup` the cosine has already come down, so the peak learning rate is never used.
  total: In the first blank, `step − total` is negative for the whole decay. In the second, `total − total` is 0 and the division fails.
  step: "`step − step` is always 0, so the learning rate would sit at max_lr forever."
why_ok: "`progress` runs from 0 at the end of warmup to 1 at `total`. The cosine then goes from 1 to −1, so `lr` goes from max_lr down to exactly min_lr."
why_wrong: "The decay starts where warmup ends. Count the steps since then, out of the steps the decay lasts."
hint: The decay starts when warmup ends. How many decay steps have passed, out of how many there are?
```

```card
type: predict
id: predict_midpoint
eyebrow: Predict the output
title: Warmup is 100 steps and the run is 1,100. What prints?
code: |
  # max_lr 2e-4, min_lr 2e-5, cosine decay after warmup
  print(f"{lr_at(600, 2e-4, 100, 1100, 2e-5):.2e}")
options:
  - text: "1.00e-04"
    why: That's halfway between max_lr and zero. The cosine decays to min_lr, not to zero, so the midpoint is halfway between 2e-4 and 2e-5.
  - text: "1.10e-04"
    correct: true
    why: "Step 600 is 500 steps into a 1,000-step decay, so progress = 0.5 and cos(π/2) = 0. That leaves min_lr + 0.5 × (max_lr − min_lr) = 2e-5 + 9e-5 = 1.1e-4."
  - text: "9.72e-05"
    why: That treats step 600 as 600 / 1100 of the way through, as if the decay started at step 0. It starts after warmup, at step 100.
  - text: "2.00e-05"
    why: That's where the decay ends, at step 1,100. Step 600 is only halfway.
hint: Work out progress first. The decay starts at step 100 and lasts 1,000 steps.
```

```card
type: quiz
ref: same_text
```

```card
type: quiz
ref: overfit_signal
```

```card
type: quiz
ref: why_warmup
```

## Build steps

### Plan the run

```challenge
id: token_plan
file: plan.py
minutes: 20
start:
  code: |
    import math

    # The control run (4A.4), from its log.
    CONTROL_TOKENS = 20_000_000   # the 20M-token sample, in Llama 3.1 tokens (seed 0, ids saved)
    CONTROL_TOK_PER_S = 790       # its throughput on a 12 GB RTX 3060
    TOKENS_PER_STEP = 4 * 2048    # its batch: 4 sequences of 2,048 tokens. Keep it.

    # Your extended run.
    FERT_LLAMA = 3.38             # tokens per Mongolian word, Llama 3.1's tokenizer (your 1.6 audit; reference 3.38)
    FERT_EXT = 1.62               # tokens per word, your extended tokenizer (4B.1)
    TOK_PER_S = 760               # from a 50-step test: the new rows and the bigger head cost a little

setup:
  - say: "The control run trained on a fixed 20 million Llama tokens. Your run trains on exactly the same documents, so the only difference is the tokenizer."
    pause: 3.8
  - say: "Those documents cost fewer tokens with the extended vocab. Fewer tokens means fewer steps and fewer GPU hours, at nearly the same speed per token."
    pause: 4.0
  - say: "Here's the stub. Your turn: count the same text in extended tokens, then the steps and hours for one pass."
    type_file: token_plan/stub.py
    pause: 1.2
tests: token_plan/tests.py
solution: token_plan/solution.py
alternatives:
  - token_plan/alternatives/via_words.py
  - token_plan/alternatives/ceil_steps.py
hints:
  nudge: The text stays the same, so the words stay the same. How many words are the control's tokens, and what do those words cost in extended tokens? Then a step eats a fixed number of tokens, and the GPU a fixed number per second.
  concept: "Words = Llama tokens / 3.38, and extended tokens = words × 1.62, so `llama_tokens * fert_ext / fert_llama`. One pass takes `tokens // tokens_per_step` whole steps. Time is tokens over tokens per second, which gives seconds; divide by 3,600 for hours."
  skeleton: |
    return llama_tokens * ________ / __________
    steps = int(______ // _______________)
    hours = ______ / _________ / ____
  solution: |
    return llama_tokens * fert_ext / fert_llama
    steps = int(tokens // tokens_per_step)
    hours = tokens / tok_per_s / 3600
on_solve: "The control took 2,441 steps and 7.0 hours of training (7.1 GPU hours with evals); the same documents take your run 1,170 steps and 3.5 hours. That's the tokenizer paying for itself before any eval: under half the steps for the same text. Add a few minutes of evals and the real run is about 3.6 GPU hours. If you want to spend the saved hours, a second run on more text is a separate experiment. Don't mix it into the comparison."
mistakes:
  - { file: token_plan/mistakes/placeholders.py, fails: same_text, says: "still returns `...`" }
  - { file: token_plan/mistakes/ratio_upside_down.py, fails: same_text, says: "upside down" }
  - { file: token_plan/mistakes/unchanged.py, fails: same_text, says: "the same number of tokens" }
  - { file: token_plan/mistakes/steps_placeholder.py, fails: steps, says: "`steps` is still `...`" }
  - { file: token_plan/mistakes/fractional_steps.py, fails: steps, says: "with a fraction" }
  - { file: token_plan/mistakes/steps_times.py, fails: steps, says: "far more steps than there are tokens" }
  - { file: token_plan/mistakes/seconds.py, fails: hours, says: "that's seconds" }
  - { file: token_plan/mistakes/minutes.py, fails: hours, says: "that's minutes" }
  - { file: token_plan/mistakes/steps_per_s.py, fails: hours, says: "not steps per second" }
```

### Write the learning-rate schedule

```challenge
id: lr_at
file: plan.py
minutes: 30
start: { from: token_plan }
setup:
  - say: "The plan gives you the number of steps. The learning rate over those steps is the other half: a short warmup, then a cosine down to a floor. The peak is the control's, 1e-4, so that stays the same too."
    pause: 4.6
  - say: "Here's the stub. It prints the schedule at a few steps, as a bar. Your turn: write lr_at."
    type_file: lr_at/stub.py
    pause: 1.2
tests: lr_at/tests.py
solution: lr_at/solution.py
alternatives:
  - lr_at/alternatives/warmup_from_zero.py
  - lr_at/alternatives/coeff.py
hints:
  nudge: Three pieces. A straight line up during warmup, a flat floor after the end, and in between a half cosine that starts at max_lr and lands exactly on min_lr. Where does the cosine start counting?
  concept: "Warmup climbs by `max_lr / warmup` per step, so `max_lr * (step + 1) / warmup` reaches max_lr on the last warmup step. For the decay, `progress = (step - warmup) / (total - warmup)` runs from 0 to 1. `0.5 * (1 + cos(π · progress))` runs from 1 to 0, so scale it by `max_lr - min_lr` and add `min_lr`."
  skeleton: |
    return max_lr * (____ + 1) / ______
    return ______
    progress = (step - ______) / (_____ - ______)
    return min_lr + 0.5 * (______ - ______) * (1 + math.cos(_______ * progress))
  solution: |
    return max_lr * (step + 1) / warmup
    return min_lr
    progress = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
on_solve: "That's the schedule for all 1,170 steps. 4A.4's command doesn't set a schedule, so check what your control actually ran: its log prints the learning rate. Both runs need the same schedule. If the control's was different, use the control's here, or rerun the control with this one. The new rows can take a higher peak than LoRA (Hints). On the GPU, `torch.optim.lr_scheduler.LambdaLR` with `lambda s: lr_at(s, 1.0, ...)` turns this into a scheduler."
mistakes:
  - { file: lr_at/mistakes/placeholders.py, fails: warmup_ramp, says: "still returns `...`" }
  - { file: lr_at/mistakes/no_warmup.py, fails: warmup_ramp, says: "starts at the peak" }
  - { file: lr_at/mistakes/to_zero.py, fails: cosine_decay, says: "decays to 0" }
  - { file: lr_at/mistakes/whole_run.py, fails: cosine_decay, says: "counts from step 0" }
  - { file: lr_at/mistakes/no_pi.py, fails: cosine_decay, says: "π" }
  - { file: lr_at/mistakes/full_period.py, fails: cosine_decay, says: "a full cosine" }
  - { file: lr_at/mistakes/linear.py, fails: cosine_decay, says: "straight line" }
  - { file: lr_at/mistakes/upside_down.py, fails: cosine_decay, says: "climbs" }
  - { file: lr_at/mistakes/zero_after.py, fails: after_end, says: "drops to 0" }
```

### Run continued pretraining

```replay
id: cpt_run
intro: "The real run is about 3.6 hours on the control's 12 GB GPU: 1,170 steps over the control's 20M-token sample, with held-out bits per character every 250 steps. Watch a recording first so you know what healthy looks like."
log: replays/cpt-run.jsonl
speed: 40
recorded:
  hardware: RTX 3060 12 GB
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: Held-out bits per character, every 250 steps
  flag_above: 0.67
  x_labels: [step 0, "step 1,170"]
  caption: Red bars are worse than 4A.4's control run (0.67 bits per character). The extended model passes it around step 1,000 and ends at 0.642, 4.2% better, on the same text in half the steps. Its per-token perplexity, 9.7, looks far worse than the control's 3.12. That's the tokenizer, not the model.
summary:
  - { label: Bits per character, value: 1.08 → 0.642 (control 0.67) }
  - { label: vs control, value: −4.2% }
  - { label: GPU hours, value: 3.6 (control 7.1) }
run_yourself: |
  # on your GPU, from the course repo
  python continued_pretrain.py \
    --model runs/llama31-mn-8k-init --tokenizer tokenizers/llama31-mn-8k \
    --data data/mn-clean/train.jsonl --sample-ids runs/control/sample_ids.txt \
    --rank 32 --alpha 64 --lr 1e-4 --min-lr 1e-5 --warmup 50 --seq 2048 --grad-accum 4 \
    --train-new-rows --new-rows-lr 5e-4 --eval-every 250 --out runs/llama31-mn-8k-cpt
```

### Record your metrics

```metrics
id: record
intro: Bits per character of your control run and of your extended run on the same held-out split, and the GPU hours this run took. They're compared with the reference run and the pass limit.
```

## Hints

- Load the control's `sample_ids.txt` and train on exactly those documents. If you sample again, even with the same seed after a corpus change, the comparison is off.
- Train only the new rows of `embed_tokens` and `lm_head`: keep them as a separate 8,000-row `nn.Parameter` and concatenate with the frozen old rows in `forward`. A gradient hook that zeroes rows 0…128,255 looks simpler, but it still allocates gradients and Adam state for all 136,256 rows of both matrices (about 13.4 GB in fp32 for 1.1B parameters: gradient plus two Adam moments, 12 bytes each, more than a 12 GB card has), and weight decay still moves the "frozen" rows.
- New rows usually want a higher learning rate than LoRA, around 5× (here 5e-4 against 1e-4), in their own optimizer param group. Chinese LLaMA went further and trained the embeddings alone for a first stage.
- Bits per character = total held-out loss in nats / ln 2 / characters, pooled over the whole split, with 4A.4's eval code. Per-token perplexity is only for watching one run's curve.
- Keep the control's English check (6.10 → 6.41 there). If the extended run forgets much more English, look at the new-rows learning rate first.
- Save the best held-out checkpoint, not just the last one. 4B.4 starts from it.

## Done when

- [ ] Steps and GPU hours planned for the control's text {#run_planned auto=challenge:token_plan}
- [ ] Warmup and cosine schedule pass the tests {#schedule_written auto=challenge:lr_at}
- [ ] Continued pretraining finished on the control's sample, with held-out bits per character below your 4A.4 control run {#cpt_done}
- [ ] Loss vs control and GPU hours recorded {#metrics_recorded auto=metrics}
