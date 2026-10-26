---
id: "4A.4"
title: Control run, untouched tokenizer
summary: Continue pretraining Llama 3.1 8B on your Mongolian corpus with its own tokenizer, and measure held-out perplexity the way 4B will need to compare against it.
phase: "4A"
hours: 6
type: run
status: ready
protected: true
cut: null
prerequisites:
  - 4A.3
papers:
  - title: "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks"
    authors: Gururangan et al.
    year: 2020
    url: https://arxiv.org/abs/2004.10964
quality_metric:
  id: mn_ppl
  label: Mongolian held-out perplexity
  unit: ppl
  note: Per token, with Llama 3.1's tokenizer, on 4A.3's held-out split. The base model scores about 5.8. Write down bits per character too, because 4B compares with that.
  inputs:
    - { id: mn_ppl, label: Held-out perplexity after the run, hint: The last eval line }
  compute: value
  reference: 3.12
  pass: { max: 4.0 }
systems_metric:
  id: gpu_hours
  label: GPU hours
  unit: h
  note: Training plus evals, one epoch over a 20M-token sample. 4B's runs use the same budget, so keep it.
  inputs:
    - { id: gpu_hours, label: GPU hours, hint: Wall-clock time of the run }
  compute: value
  reference: 7.1
  pass: { max: 10.0 }
quiz:
  - id: why_control
    q: 4B.4 compares a Mongolian tokenizer against this run. Why not compare against base Llama 3.1 instead?
    answer: 1
    options:
      - text: Base Llama is a fine baseline. The control only saves compute later
        why: Against base Llama, the new tokenizer would get credit for everything continued pretraining does on its own. The control costs compute. It doesn't save any.
      - text: Training on Mongolian helps whatever the tokenizer is. Only a run with the same data, steps and settings isolates what the tokenizer adds
        why: That's the point of a control. Change one thing, the tokenizer, and keep everything else identical, so the difference between the runs belongs to that one thing.
      - text: Base Llama can't be evaluated on Mongolian at all
        why: It can, and it scores a held-out perplexity of about 5.8 per token. It's just the wrong comparison for 4B's question.
  - id: across_tokenizers
    q: A 4B run scores per-token perplexity 9.1 on the held-out set. Your control scored 3.1. What can you conclude?
    answer: 0
    options:
      - text: Nothing yet. Their tokens cover different amounts of text, so compare bits per character
        why: A tokenizer with longer tokens has fewer, harder predictions, and its per-token perplexity goes up even if the model is better. Bits per character divides by something both runs share, the text itself.
      - text: The 4B run is about 3× worse
        why: Per-token perplexity is only comparable when both runs cut the text into the same tokens. Here they don't.
      - text: The 4B run is better, because a higher perplexity means more information per token
        why: It might be better, but perplexity can't tell you. Convert both to bits per character and compare those.
  - id: english_forgetting
    q: After the control run, English perplexity rose from 6.10 to 6.41. What happened?
    answer: 2
    options:
      - text: The tokenizer changed and English now splits differently
        why: This run leaves the tokenizer untouched. English tokenizes exactly as before.
      - text: The eval is broken, since a frozen base can't get worse
        why: The base weights are frozen, but the adapters change every layer's output. They were trained only on Mongolian.
      - text: Mild forgetting. The adapters pulled the model toward Mongolian
        why: About +5% is typical for an all-Mongolian run. If it matters for your use, mix a few percent of English into the training data and keep the rest of the recipe the same.
---

## Goal

Run the **control**: continue pretraining Llama 3.1 8B on a 20M-token sample of your clean Mongolian corpus with QLoRA, and change nothing else. The tokenizer stays exactly as Meta shipped it. Every 4B experiment changes the tokenizer and is measured against this run, so it has to be a fair baseline: same data, steps and settings. First write the evaluation itself: held-out perplexity pooled correctly over documents, and **bits per character**, which still means something when 4B changes the tokenizer.

## Learn

```card
type: concept
id: control_run
eyebrow: Why a control
title: To credit the tokenizer, first see what Mongolian data does alone.
visual:
  kind: bars
  max: 6
  bars:
    - { label: base, value: 5.84, text: ppl 5.84 }
    - { label: control, value: 3.12, text: ppl 3.12 }
body: |
  4B builds a Mongolian tokenizer and claims it helps. Helps compared with what? Continued pretraining on Mongolian text lowers perplexity a lot on its own, whatever the tokenizer. The control does exactly that, with Llama's tokenizer untouched, so any gap that 4B opens up beyond it is the tokenizer's. The bars are held-out perplexity per token before and after the control run.
```

```card
type: predict
id: pooled
eyebrow: Predict the output
title: Two held-out documents. What's the perplexity of both together?
code: |
  # doc A: 10 predicted tokens, total loss 10 nats    (its own ppl: e^1 = 2.72)
  # doc B: 1000 predicted tokens, total loss 2000 nats (its own ppl: e^2 = 7.39)
  print(round(math.exp((10 + 2000) / (10 + 1000)), 2))
options:
  - text: "5.05"
    why: That's the average of the two perplexities. It counts a 10-token document as much as a 1000-token one.
  - text: "4.48"
    why: "That's e to the average of the two mean losses, e^1.5. Same problem: each document gets equal weight whatever its length."
  - text: "7.32"
    correct: true
    why: 2010 nats over 1010 tokens is 1.99 nats per token, and e^1.99 = 7.32. Pooled, every token counts once, so the long document dominates, as it should.
hint: Add the losses, add the tokens, divide once.
```

```card
type: calc
id: bpc_calc
eyebrow: Do the math
title: How many bits per character?
given:
  - { label: Total held-out loss, value: "1,000 nats" }
  - { label: Characters in the held-out text, value: "2,000" }
  - { label: One bit, value: ln 2 ≈ 0.693 nats }
unit: bits/char
answer: { min: 0.72, max: 0.73 }
placeholder: "0.00"
why_ok: 1000 / 0.693 = 1443 bits, over 2000 characters is 0.72 bits per character. Neither number depends on how a tokenizer cut the text, which is what makes it fair across 4B's runs.
why_wrong: Turn nats into bits by dividing by ln 2, then divide by the number of characters.
hint: Bits = nats / ln 2.
```

```card
type: fill_blank
id: next_token
eyebrow: Fill the blanks
title: Sum the next-token loss of one document.
code: |
  def doc_nll(logits, ids):
      pred = logits[[[0]]]
      target = ids[[[1]]]
      return F.cross_entropy(pred, target, reduction=[[2]])
tokens: [":-1", "1:", ":", '"sum"', '"mean"']
answer: [":-1", "1:", '"sum"']
tips:
  ":": "Without the shift, position t is scored on the token it was given as input. A model that copies its input would look perfect."
  '"mean"': "A per-document average can't be pooled: a 10-token document would count as much as a 1000-token one. Sum here, divide once at the end."
why_ok: "The last position has nothing after it to predict and the first token (BOS) is never predicted, so the logits drop their last row and the ids their first. Summing lets heldout pool documents token by token."
why_wrong: "Position t predicts the token at t + 1. Which rows of logits have a next token, and which ids are ever predicted?"
hint: The first token is never predicted. The last position predicts nothing.
```

```card
type: quiz
ref: why_control
```

```card
type: quiz
ref: across_tokenizers
```

```card
type: quiz
ref: english_forgetting
```

## Build steps

### Measure held-out perplexity and bits per character

```challenge
id: heldout_eval
file: heldout.py
minutes: 35
start: { code: "import math\nimport torch\nimport torch.nn.functional as F\n\n" }
setup:
  - say: "Three held-out Mongolian sentences, and logits for them. The numbers are stand-ins, shaped like your eval's: a cut-down vocabulary, BOS first, and position t scores the token at t + 1."
    type_file: heldout_eval/data.py
    pause: 3.2
  - say: "CONTROL is your control run, cutting Mongolian into short pieces with Llama's tokenizer. PREVIEW_4B is the same text through a Mongolian tokenizer, the kind you'll train in 4B."
    pause: 3.6
  - say: "Here's the stub. Your turn: the loss of one document, then perplexity and bits per character over all of them."
    type_file: heldout_eval/stub.py
    pause: 1.2
tests: heldout_eval/tests.py
solution: heldout_eval/solution.py
alternatives:
  - heldout_eval/alternatives/log_softmax.py
  - heldout_eval/alternatives/tensor_sum.py
  - heldout_eval/alternatives/loop.py
  - heldout_eval/alternatives/log2.py
hints:
  start:
    try: |
      text, logits, ids = CONTROL[0]
      print(len(ids), len(text), logits.shape)
      print(round(F.cross_entropy(logits[0:1], ids[1:2]).item(), 3))
    see: |
      33 76 torch.Size([33, 64])
      1.494
    next: "That's one prediction: row 0 of the logits scored against the id that comes after it, `ids[1]`, a loss of about 1.49. This document has 33 ids, so 32 predictions, and `doc_nll` needs their total. Slice both sides so row t meets id t + 1, and ask the loss to add them up rather than average them."
  logic: |
    doc_nll:
      each position predicts the next token
        → the last row has no next token, the first id is never predicted
      score every prediction's loss, add them up
        → return the total
    heldout:
      for each document:
        add its loss to total_nll
        add its predicted tokens to n_tokens
        add its characters to n_chars
      ppl = e to the power (total_nll ÷ n_tokens)
      bpc = total_nll in bits (÷ ln 2) ÷ n_chars
        → return ppl and bpc
  tools: |
    line up predictions and targets
      x[a:b]                       slice of rows or ids
    loss of every prediction
      F.cross_entropy(logits, y, reduction=m)
                                   loss; m picks how to combine
      t.item()                     one-value tensor to a number
    pool over the documents
      sum(expr for d in docs)      adds one value per document
      len(x)                       how many items are in x
    perplexity and bits per character
      math.exp(x)                  e to the power x
      math.log(x)                  natural log
  nudge: "`doc_nll`: shift by one and sum. `heldout`: add everything up first, the loss, the predicted tokens and the characters, then divide once."
  concept: "`F.cross_entropy(logits[:-1], ids[1:], reduction=\"sum\")` scores every position against the token after it. A document of n ids makes n − 1 predictions, because BOS is never predicted. Perplexity is `exp(total loss / predicted tokens)`. Bits per character is `total loss / ln 2 / characters`: nats to bits, then per character, which doesn't depend on the tokenizer."
  example: |
    Use a vocabulary of 4 and two tiny documents.

    - Doc A: `ids` has 3 entries, so 2 predictions. All-zero logits give each token probability 1/4, a loss of ln 4 ≈ 1.386 each. `doc_nll` = 2 × 1.386 ≈ `2.773`. Its text is 8 characters.
    - Doc B: `ids` has 5 entries, so 4 predictions. Suppose each correct token gets probability 1/2: a loss of ln 2 ≈ 0.693 each. `doc_nll` = 4 × 0.693 ≈ `2.773`. Its text is 12 characters.
    - Pooled: `total_nll` ≈ 5.545, `n_tokens` = 2 + 4 = `6`, `n_chars` = `20`.
    - `ppl` = exp(5.545 / 6) ≈ `2.52`.
    - Bits: 5.545 / ln 2 = 8 bits. `bpc` = 8 / 20 = `0.4`.

    Check: A alone has perplexity 4 and B alone has 2. Their plain average is 3, which is not 2.52. Pooling first weights B for its extra predictions.
  skeleton: |
    return F.cross_entropy(logits[____], ids[___], reduction="___").item()
    total_nll = sum(doc_nll(logits, ids) for _, logits, ids in docs)
    n_tokens = sum(len(ids) - ___ for _, _, ids in docs)
    n_chars = sum(___(text) for text, _, _ in docs)
    ppl = math.___(total_nll / n_tokens)
    bpc = total_nll / math.log(___) / ________
  solution: |
    return F.cross_entropy(logits[:-1], ids[1:], reduction="sum").item()
    total_nll = sum(doc_nll(logits, ids) for _, logits, ids in docs)
    n_tokens = sum(len(ids) - 1 for _, _, ids in docs)
    n_chars = sum(len(text) for text, _, _ in docs)
    ppl = math.exp(total_nll / n_tokens)
    bpc = total_nll / math.log(2) / n_chars
on_solve: "Per token, the 4B preview looks four times worse, 13.6 against 3.3. Per character it's slightly better, 0.69 bits against 0.73. Its tokens are longer, so each one carries more text and is harder to guess. Per-token perplexity only compares runs that share a tokenizer. Bits per character compares anything, which is why 4B.4 reports it."
mistakes:
  - { file: heldout_eval/mistakes/placeholders.py, fails: doc_nll, says: "`doc_nll` still returns `...`" }
  - { file: heldout_eval/mistakes/no_shift.py, fails: doc_nll, says: "Position t predicts the *next* token" }
  - { file: heldout_eval/mistakes/shift_backwards.py, fails: doc_nll, says: "asking each position for the token *before* it" }
  - { file: heldout_eval/mistakes/mean_reduction.py, fails: doc_nll, says: "You return the *average* loss per token" }
  - { file: heldout_eval/mistakes/log_likelihood.py, fails: doc_nll, says: "that's the log-likelihood" }
  - { file: heldout_eval/mistakes/ppl_placeholder.py, fails: pooled_ppl, says: "`ppl` is still `...`" }
  - { file: heldout_eval/mistakes/mean_of_ppl.py, fails: pooled_ppl, says: "You average the three documents' perplexities" }
  - { file: heldout_eval/mistakes/mean_of_means.py, fails: pooled_ppl, says: "You average each document's mean loss" }
  - { file: heldout_eval/mistakes/counts_bos.py, fails: pooled_ppl, says: "BOS included" }
  - { file: heldout_eval/mistakes/no_exp.py, fails: pooled_ppl, says: "Perplexity is e to that power" }
  - { file: heldout_eval/mistakes/base_two.py, fails: pooled_ppl, says: "Use e, `math.exp`" }
  - { file: heldout_eval/mistakes/bpc_nats.py, fails: bits_per_char, says: "That's nats per character" }
  - { file: heldout_eval/mistakes/bpc_per_token.py, fails: bits_per_char, says: "That's bits per *token*" }
  - { file: heldout_eval/mistakes/bpc_bytes.py, fails: bits_per_char, says: "That's bits per *byte*" }
  - { file: heldout_eval/mistakes/bpc_times_ln2.py, fails: bits_per_char, says: "nats become bits by *dividing* by ln 2" }
```

### Run the control

```replay
id: control_run
intro: "QLoRA continued pretraining of Llama 3.1 8B on 20M tokens of your clean corpus, one epoch, about 7 hours on a 12 GB GPU. The tokenizer and the embeddings stay frozen. Only the adapters learn. Watch a recording first so you know what healthy looks like."
log: replays/control-run.jsonl
speed: 40
recorded:
  hardware: RTX 3060 12 GB
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: Held-out Mongolian perplexity at each eval
  flag_above: 4.0
  x_labels: [step 0, step 2441]
  caption: "Red bars are above 4.0. Two thirds of the drop comes in the first 500 steps, the rest slowly. Bits per character fall from 1.04 to 0.67, the number 4B's runs have to beat."
summary:
  - { label: Held-out ppl, value: 5.84 → 3.12 }
  - { label: Bits per character, value: 1.04 → 0.67 }
  - { label: English ppl, value: 6.10 → 6.41 }
  - { label: GPU hours, value: "7.1" }
run_yourself: |
  # on your machine or Colab, from the course repo
  python continued_pretrain.py \
    --model meta-llama/Llama-3.1-8B --data data/mn-clean/train.jsonl --tokens 20M \
    --rank 32 --alpha 64 --lr 1e-4 --seq 2048 --grad-accum 4 \
    --eval data/mn-clean/heldout.jsonl --eval-every 250 --out runs/control
```

### Record your metrics

```metrics
id: record
intro: Held-out perplexity after your control run, and its GPU hours. They're compared with the reference run and the pass limits. Note the bits per character beside them; 4B.4 compares against it.
```

## Hints

- Fix the 20M-token sample with a seed and save its document ids. 4B's runs must train on exactly the same text, or the comparison stops being fair.
- Evaluate the base model first, step 0, with the same script. If the base number looks wrong, the eval is wrong, and you've lost nothing yet.
- Pack documents into 2048-token sequences for training, but evaluate each held-out document on its own so no document is scored with another one's context.
- Save the adapter every 500 steps. A 7-hour run on a desktop GPU will meet a power cut or a driver update eventually.
- Keep the English eval in the loop. A big rise there means the learning rate or the rank is too aggressive for continued pretraining.

## Done when

- [ ] Your eval pools tokens correctly and reports bits per character {#heldout_eval auto=challenge:heldout_eval}
- [ ] The control run finishes one epoch, and its adapter and eval log are saved {#control_saved}
- [ ] Held-out perplexity and GPU hours recorded {#metrics_recorded auto=metrics}
