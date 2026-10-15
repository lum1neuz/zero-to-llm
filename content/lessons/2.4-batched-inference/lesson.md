---
id: "2.4"
title: Batched inference
summary: Run many prompts at once. Pad them on the left, give each row its own positions and an attention mask, and get exactly what each prompt gives alone.
phase: "2"
hours: 4
type: build
status: ready
protected: false
cut: 2
prerequisites:
  - "2.2"
papers: []
quality_metric:
  id: batch_match
  label: Batched outputs match single-sequence outputs
  unit: "%"
  inputs:
    - { id: match_pct, label: Prompts whose batched greedy output matches one-at-a-time (%), hint: "64 prompts, batch 16, fp32" }
  compute: value
  reference: 100
  pass: { min: 95 }
systems_metric:
  id: throughput_x
  label: Throughput gain at batch 16
  unit: ×
  note: Real new tokens per second at batch 16 divided by batch 1, KV cache on, same prompts.
  inputs:
    - { id: tok_s_b1, label: tok/s at batch 1, hint: From generate.py --batch-sizes }
    - { id: tok_s_b16, label: tok/s at batch 16, hint: Same run }
  compute: ratio
  reference: 11.2
  pass: { min: 4 }
quiz:
  - id: pad_keys
    q: Why must real tokens never attend to pad tokens?
    answer: 0
    options:
      - text: A pad is a real embedding vector, so attending to it changes the output, and a padded prompt must give what it gives alone
        why: The pad id looks up an ordinary row of the embedding table. Without the mask, a short prompt's output depends on how many pads it got, which is the bug the matches_single test catches.
      - text: Pad tokens are NaN
        why: They're ordinary ids with ordinary embeddings. That's exactly why they leak into the output when they aren't masked.
      - text: Only for speed. The output is the same either way
        why: Masking costs the same as not masking. It's for correctness. Your test showed logits off by more than 1 without it.
  - id: why_left
    q: Why pad on the left for generation, not the right?
    answer: 1
    options:
      - text: Right padding needs a bigger attention mask
        why: The mask is [B, T, T] either way. The problem with right padding is where the last real token sits.
      - text: So every row ends on a real token, and each step's new ids all go in the same next column
        why: Generation reads `logits[:, -1]`. With right padding, the shorter rows' last column is a pad, so their next token would be predicted after a pad.
      - text: Left padding is faster on GPUs
        why: Speed is the same. It's about every row's last column being a real token.
  - id: finfo_min
    q: The batched model fills masked scores with torch.finfo(dtype).min instead of -inf. Why?
    answer: 2
    options:
      - text: It's faster to compute
        why: Both are a single masked_fill. The difference is what happens on a row with nothing to attend to.
      - text: -inf doesn't work with masked_fill
        why: masked_fill takes -inf fine. Lesson 2.1's model uses it.
      - text: A left pad's query has no allowed keys at all, and softmax over a row of -inf gives NaN, which can leak into later layers
        why: With the most negative float, that row gets a harmless uniform average. Real rows are unchanged, because exp(min − max) is exactly 0 in float32.
---

## Goal

Generate for many prompts in one forward pass. Prompts have different lengths, so **left-pad** them into one `[B, T]` tensor and keep a mask of which tokens are real. Then give every row its own **positions**, counted from its first real token, and an **attention mask** that's causal and hides the pads. The test is strict: each row must produce exactly what its prompt produces alone. Then measure the throughput gain on your 30M model.

## Learn

```card
type: concept
id: why_batch
eyebrow: Why batch
title: One read of the weights, sixteen tokens.
visual:
  kind: bars
  max: 12000
  bars:
    - { label: batch 1, value: 480, text: "480 tok/s" }
    - { label: batch 4, value: 1820, text: "1,820" }
    - { label: batch 16, value: 5380, text: "5,380" }
    - { label: batch 64, value: 11900, text: "11,900" }
body: |
  At batch 1, each decode step reads every weight from memory to make one token, and the GPU spends most of its time waiting on that read. A batch of B prompts reuses the same read for B tokens, so a step costs about the same and makes B times as many. The catch: prompts have different lengths, and a batch must be a rectangle.
```

```card
type: predict
id: predict_left_pad
eyebrow: Predict the output
title: What does left_pad build?
code: |
  prompts = [torch.tensor([5, 6, 7]), torch.tensor([8])]
  idx, mask = left_pad(prompts, pad_id=0)
  print(idx.tolist())
options:
  - text: "[[5, 6, 7], [0, 0, 8]]"
    correct: true
    why: The shorter prompt gets two pads in front, so both rows end on a real token in the last column. mask is [[T, T, T], [F, F, T]].
  - text: "[[5, 6, 7], [8, 0, 0]]"
    why: That's right padding. Row 1's last column is a pad, so generation would read the prediction made after a pad.
  - text: "[[5, 6, 7], [8]]"
    why: A tensor has to be a rectangle. Ragged rows are what padding fixes.
  - text: "[[0, 5, 6, 7], [0, 0, 0, 8]]"
    why: The batch is as wide as the longest prompt, 3. The longest prompt needs no pads.
hint: The batch is as wide as the longest prompt. Shorter prompts get pads on one side, so that every row ends on a real token.
```

```card
type: fill_blank
id: position_blank
eyebrow: Fill the blanks
title: Count positions from each row's first real token.
code: |
  # mask = [[False, False, True, True], [True, True, True, True]]
  pos = (mask.[[0]](dim=-1) - 1).clamp(min=[[1]])
  # pos  = [[0, 0, 0, 1], [0, 1, 2, 3]]
tokens: [cumsum, sum, "0", "-1", argmax, "1"]
answer: [cumsum, "0"]
tips:
  sum: "`sum` gives one number per row, the count of real tokens. You need a running count, one per position."
  "-1": "That leaves the pads at −1, and the position embedding can't look up −1."
  argmax: argmax gives one index per row, not a position for every token.
  "1": That pushes the first real token to position 1. It should be 0.
why_ok: "The running count of real tokens is 1, 2, 3… on the real tokens, so minus 1 gives 0, 1, 2…. Pads come out as −1, and clamp lifts them to 0. Their position doesn't matter, because nothing attends to them."
why_wrong: "You need a running count of real tokens, minus 1, with the pads kept at a valid index."
hint: Treat True as 1 and False as 0. Which running total gives 1, 2, 3 on the real tokens?
```

```card
type: calc
id: padding_waste
eyebrow: Do the math
title: What share of this batch's prefill is padding?
given:
  - { label: Prompt lengths, value: "12, 40, 7, 21 tokens" }
  - { label: Padded to, value: the longest prompt }
unit: "%"
answer: { min: 49, max: 51 }
placeholder: "0"
why_ok: "4 rows × 40 = 160 slots, and 12 + 40 + 7 + 21 = 80 are real, so 80 of 160, 50%, is padding. Sorting prompts by length before batching cuts this. Continuous batching (5.4) avoids it."
why_wrong: Every row is padded to 40 tokens. Count all the slots, subtract the real tokens, and divide by all the slots.
hint: "Total slots: 4 rows × 40. Real tokens: 12 + 40 + 7 + 21."
```

```card
type: quiz
ref: pad_keys
```

```card
type: quiz
ref: why_left
```

```card
type: quiz
ref: finfo_min
```

## Build steps

### Pad on the left

```challenge
id: left_pad
file: batched.py
minutes: 20
start:
  code: |+
    import torch
    import torch.nn.functional as F

    # 2.1's tiny model, now batched: ids [B, T] in, logits [B, T, V] out.
    # pos says where each token sits, and keep says which keys each query may attend to.
    text = "the cat sat on the mat. the dog sat on the log. a cat and a dog ran to the mat. "
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    encode = lambda s: torch.tensor([stoi[c] for c in s])
    decode = lambda ids: "".join(chars[i] for i in ids.tolist())


    class TinyLM:
        def __init__(self, seed=0):
            g = torch.Generator().manual_seed(seed)
            V, C, CTX = len(chars), 16, 64
            self.C = C
            counts = torch.ones(V, V) * 0.1
            for a, b in zip(text, text[1:]):
                counts[stoi[a], stoi[b]] += 1
            self.bigram = (counts / counts.sum(dim=1, keepdim=True)).log()
            self.wte = torch.randn(V, C, generator=g)
            self.wpe = torch.randn(CTX, C, generator=g) * 0.5
            self.Wq = torch.randn(C, C, generator=g) / C**0.5
            self.Wk = torch.randn(C, C, generator=g) / C**0.5
            self.Wv = torch.randn(C, C, generator=g) / C**0.5
            self.Wo = torch.randn(C, V, generator=g) * 0.15

        def __call__(self, idx, pos=None, keep=None):
            B, T = idx.shape
            if pos is None:   # no padding: every row counts 0, 1, 2, ...
                pos = torch.arange(T).expand(B, T)
            if keep is None:  # no padding: plain causal mask
                keep = torch.tril(torch.ones(T, T, dtype=torch.bool)).expand(B, T, T)
            x = F.embedding(idx, self.wte) + F.embedding(pos, self.wpe)
            q, k, v = x @ self.Wq, x @ self.Wk, x @ self.Wv
            att = (q @ k.transpose(1, 2)) / self.C**0.5
            # The most negative float, not -inf: a row with no allowed keys gets junk instead of NaN
            att = att.masked_fill(~keep, torch.finfo(att.dtype).min)
            h = F.softmax(att, dim=-1) @ v
            return self.bigram[idx] + h @ self.Wo


    def generate(model, idx, max_new_tokens):
        # 2.1's greedy loop, for one unpadded sequence of shape [1, T]
        for _ in range(max_new_tokens):
            next_id = model(idx)[:, -1].argmax(dim=-1)
            idx = torch.cat([idx, next_id[:, None]], dim=1)
        return idx


    model = TinyLM()

setup:
  - say: This is 2.1's tiny model, batched. It takes ids of shape B by T, plus pos, where each token sits, and keep, which keys each query may attend to. Here are three prompts of different lengths.
    type: "prompts = [encode(\"the cat\"), encode(\"a dog\"), encode(\"the\")]\n\n"
    pause: 3.0
  - say: "A batch must be a rectangle, so the shorter prompts need padding. Put the pads in front, so every row ends on a real token, and record which tokens are real."
    pause: 3.4
  - say: "Here's the stub. Your turn: fill in the two lines in the loop."
    type_file: left_pad/stub.py
    pause: 1.2
tests: left_pad/tests.py
solution: left_pad/solution.py
alternatives:
  - left_pad/alternatives/negative_slice.py
  - left_pad/alternatives/start_var.py
hints:
  nudge: Row b is T wide and prompt p is len(p) long. Which column does p have to start in so that it ends at the last column?
  concept: "Prompt p starts at column `T - len(p)` and runs to the end, so slice `idx[b, T - len(p):]` and assign p to it. Mark the same columns True in `mask`. Mark them by position, never by checking the ids for the pad id: a real token can share that id."
  skeleton: |
    idx[b, _ - ______:] = p
    mask[b, T - len(p)__] = ____
  solution: |
    idx[b, T - len(p):] = p
    mask[b, T - len(p):] = True
on_solve: "Every row ends on a real token. Notice the 0s inside \"the cat\" and \"a dog\": the space has id 0 too, which is why the mask records positions instead of guessing from the ids. Step 2 makes the model ignore the pads."
mistakes:
  - { file: left_pad/mistakes/placeholders.py, fails: shapes, says: "nothing was written" }
  - { file: left_pad/mistakes/right_pad.py, fails: left_padded, says: "Which side should the pads go on?" }
  - { file: left_pad/mistakes/mask_missing.py, fails: mask_marks_real, says: "`mask` is all False" }
  - { file: left_pad/mistakes/mask_inverted.py, fails: mask_marks_real, says: "True on the pads and False on the prompts" }
  - { file: left_pad/mistakes/mask_right.py, fails: mask_marks_real, says: "True exactly where the prompt sits" }
  - { file: left_pad/mistakes/mask_from_ids.py, fails: mask_marks_real, says: "the same id as the pad" }
  - { file: left_pad/mistakes/no_main.py, fails: script_runs, says: "are missing" }
```

### Positions and the attention mask

```challenge
id: batch_mask
file: batched.py
minutes: 35
start: { from: left_pad }
setup:
  - say: "Two things would make a padded row differ from its prompt alone. Its real tokens would get shifted positions, and they could attend to the pads."
    type_file: batch_mask/stub.py
    pause: 2.6
  - say: "generate_batched calls your batch_inputs every step, and the loop checks each row against the same prompt run alone. Your turn: pos and keep."
    pause: 1.2
tests: batch_mask/tests.py
solution: batch_mask/solution.py
alternatives:
  - batch_mask/alternatives/where.py
  - batch_mask/alternatives/expand.py
hints:
  nudge: "For pos: what running count is 1, 2, 3 on the real tokens and 0 on the leading pads? For keep: two conditions must both hold. What shape is each, and how do they broadcast to [B, T, T]?"
  concept: "`mask.cumsum(dim=-1)` counts real tokens as it goes. Subtract 1 so the first real token is 0, and clamp the pads up to 0. For keep, `causal` is [T, T] (query, key) and `mask[:, None, :]` is [B, 1, T]: one row of key flags per sequence, the same for every query. Combine them with `&`."
  skeleton: |
    pos = (mask.______(dim=-1) - _).clamp(min=_)
    keep = causal _ mask[:, ____, :]
  solution: |
    pos = (mask.cumsum(dim=-1) - 1).clamp(min=0)
    keep = causal & mask[:, None, :]
on_solve: "All three rows write exactly what their prompts write alone. That's the whole contract of batching: the batch changes the speed, never the answer. Step 3 measures the speed."
mistakes:
  - { file: batch_mask/mistakes/placeholders.py, fails: positions, says: "`pos` is still `...`" }
  - { file: batch_mask/mistakes/keep_placeholder.py, fails: positions, says: "`keep` is still `...`" }
  - { file: batch_mask/mistakes/arange_pos.py, fails: positions, says: "got position 2 instead of 0" }
  - { file: batch_mask/mistakes/no_clamp.py, fails: positions, says: "the pads got negative positions" }
  - { file: batch_mask/mistakes/starts_at_one.py, fails: positions, says: "they start at 1" }
  - { file: batch_mask/mistakes/shared_mask.py, fails: attention_mask, says: "one mask shared by every row" }
  - { file: batch_mask/mistakes/no_causal.py, fails: attention_mask, says: "which comes after it" }
  - { file: batch_mask/mistakes/or_instead.py, fails: attention_mask, says: "which comes after it" }
  - { file: batch_mask/mistakes/causal_only.py, fails: attention_mask, says: "the plain causal mask" }
  - { file: batch_mask/mistakes/query_mask.py, fails: attention_mask, says: "hides the pad *queries* (rows)" }
  - { file: batch_mask/mistakes/no_main.py, fails: script_runs, says: "is missing" }
```

### Batch your 30M model

```replay
id: batch_30m
intro: On your GPU, run 64 prompts through your 30M model at batch sizes 1 to 64, with the KV cache on. Watch a recording first.
log: replays/batch-30m.jsonl
speed: 40
recorded:
  hardware: RTX 3090
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: tokens per second by batch size
  flag_above: 100000
  x_labels: [batch 1, batch 64]
  caption: Up to batch 16 a step costs about the same, so throughput grows almost in step with the batch. By 64 the GPU starts doing real work and the step time rises.
summary:
  - { label: Throughput, value: "480 → 5,380 tok/s at batch 16" }
  - { label: Gain, value: 11.2× }
  - { label: Batched = single, value: 64 / 64 prompts }
run_yourself: |
  # on your machine or Colab, from the course repo
  python generate.py --ckpt runs/tinystories-30m/best.pt \
    --prompts prompts/64-openings.txt --new-tokens 256 \
    --temperature 0 --batch-sizes 1,4,16,64
```

### Record your metrics

```metrics
id: record
intro: The match rate and both throughputs come from your own run. The gain is batch 16 divided by batch 1.
```

## Hints

- With a KV cache, batching works the same way. Each row's cache has pads in front, the padding mask stays in `keep`, and new tokens append one column for every row.
- Sort prompts by length before you batch them. Similar lengths mean fewer pads and less wasted prefill.
- A finished row (one that produced end-of-text) keeps generating unless you stop it. Track a `done` flag per row and ignore its extra tokens, or remove it from the batch. Continuous batching (5.4) does this properly.
- If batched and single outputs differ by tiny amounts in fp16 or bf16, that's reduction order, not a bug. Check in fp32 first.

## Done when

- [ ] left_pad builds the batch and its mask {#left_pad auto=challenge:left_pad}
- [ ] Every padded row writes exactly what its prompt writes alone {#batch_mask auto=challenge:batch_mask}
- [ ] Batch match rate and throughput gain recorded {#metrics_recorded auto=metrics}
