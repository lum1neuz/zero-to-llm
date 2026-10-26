---
id: "4B.2"
title: Vocab extension and embedding init
summary: Grow Llama 3.1's embedding and LM head by 8,000 rows, start each new row from its old pieces, and check the loss at step 0 before you spend a GPU hour.
phase: "4B"
hours: 12
type: research
status: ready
protected: false
cut: null
prerequisites:
  - 4B.1
papers:
  - title: Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca
    authors: Cui et al.
    year: 2023
    url: https://arxiv.org/abs/2304.08177
  - title: Initializing New Word Embeddings for Pretrained Language Models
    authors: John Hewitt
    year: 2021
    url: https://nlp.stanford.edu/~johnhew/vocab-expansion.html
quality_metric:
  id: init_loss
  label: Loss at step 0 after extension
  unit: nats
  note: Mean next-token loss of the extended model on Mongolian held-out text before any training, in nats per extended-vocab token.
  inputs:
    - { id: init_loss, label: Step-0 loss (nats per token), hint: Extended model on 4A.3 held-out text before training }
  compute: value
  reference: 3.83
  pass: { max: 5.0 }
systems_metric:
  id: vocab_size
  label: Extended vocab size
  unit: tokens
  note: Llama 3.1's 128,256 plus your new tokens. A multiple of 64 keeps the LM-head matmul aligned.
  inputs:
    - { id: vocab_size, label: Rows in embed_tokens after resizing, hint: "model.get_input_embeddings().weight.shape[0]" }
  compute: value
  reference: 136256
  pass: { max: 144384 }
quiz:
  - id: random_rows_hurt
    q: You give the 8,000 new LM-head rows random values. Why does the loss go up even on text that uses only old tokens?
    answer: 0
    options:
      - text: Softmax spreads probability over every row, so random new logits steal it from the right old token
        why: Right. Each new row adds a logit to every prediction. Random rows give some of them large logits, and every bit of probability they take comes out of the correct token's share.
      - text: The old rows get overwritten when the matrix is resized
        why: Resizing copies the old rows unchanged. The model's old predictions are still there; the new logits next to them are the problem.
      - text: It doesn't. Old-token text never touches the new rows
        why: The embedding rows only matter when a new token appears in the input, but the LM head scores every token in the vocab at every position, new ones included.
  - id: append_not_prepend
    q: You build the new embedding as `torch.cat([new_rows, E])` instead of `torch.cat([E, new_rows])`. What happens?
    answer: 2
    options:
      - text: Nothing, as long as the tokenizer gives the new tokens the right ids
        why: The tokenizer gives new tokens ids 128,256 and up, after the old ones. With the new rows first, row 128,256 is an old token's row, and every old id points 8,000 rows too early.
      - text: Only the new tokens get the wrong rows
        why: It's worse than that. Every old id, 0 to 128,255, now reads a row 8,000 places away from its own.
      - text: Every old token now reads the wrong row, so the pretrained model is scrambled
        why: Right. Old ids keep their numbers, so the old rows have to stay at those numbers. New rows go after them.
  - id: per_word_loss
    q: Llama 3.1 scores 1.77 nats per token on Mongolian, and a Llama token covers 2.45 characters. The extended model starts at 3.83 nats per token, and its tokens cover 5.11 characters. Is the extended model much worse?
    answer: 1
    options:
      - text: Yes, 3.83 nats is more than twice 1.77
        why: Those are per-token numbers for two different tokenizers. An extended token carries about twice as much text as a Llama token, so its loss is naturally higher.
      - text: Only slightly. Per character it's 1.08 bits against 1.04, before any training
        why: "Right. Bits per character = nats per token / ln 2 / characters per token: 3.83 / 0.693 / 5.11 ≈ 1.08 and 1.77 / 0.693 / 2.45 ≈ 1.04. About 4% worse, with 8,000 rows that have never been trained. That's 4A.4's measure, and it works across tokenizers."
      - text: You can't tell without training first
        why: You can tell right now. Convert both to bits per character, which doesn't depend on the tokenizer.
---

## Goal

4B.1 gave you 8,000 new Mongolian tokens. Now the model needs a row for each of them in two matrices: the **embedding** (`embed_tokens`, what the model reads) and the **LM head** (`lm_head`, what it predicts). Llama 3.1 8B keeps them as two separate `[128256, 4096]` matrices. You'll grow both, start each new row as the **mean of the rows of the base pieces** the token used to split into, and check the **step-0 loss**: the loss on Mongolian text before any training. A good initialisation starts near the old model; a careless one starts worse than guessing.

## Learn

```card
type: concept
id: what_grows
eyebrow: What changes
title: Two matrices grow by 8,000 rows. Nothing else in the model changes.
visual:
  kind: bars
  max: 7
  limit: { value: 4.80, label: uniform guess }
  bars:
    - { label: random rows, value: 6.39, text: "6.39" }
    - { label: zero rows, value: 4.08, text: "4.08" }
    - { label: mean of all rows, value: 3.89, text: "3.89" }
    - { label: mean of pieces, value: 3.23, text: "3.23" }
body: |
  `embed_tokens` goes from `[128256, 4096]` to `[136256, 4096]`, and `lm_head` the same. Rows 0 to 128,255 stay exactly as they are, because the old tokens keep their ids. The 8,000 new rows need starting values. The bars are step-0 losses (nats per token) of the toy model you'll build: random rows are worse than a uniform guess over the vocab. Starting each new token from the mean of its old pieces (байна from ба, й and на) starts closest to what the model already knows.
```

```card
type: predict
id: predict_cat
eyebrow: Predict the output
title: What prints?
code: |
  E = torch.ones(4, 3)
  new_rows = torch.zeros(2, 3)
  print(torch.cat([E, new_rows], dim=0).shape)
options:
  - text: torch.Size([6, 3])
    correct: true
    why: "`dim=0` stacks rows: 4 old rows, then 2 new ones. Each row is still 3 wide. That's exactly a vocab growing from 4 to 6 tokens."
  - text: torch.Size([4, 5])
    why: That would be joining along `dim=1`, and it can't work here. `cat` along dim 1 needs both to have 4 rows, and `new_rows` has 2.
  - text: torch.Size([2, 4, 3])
    why: That's `torch.stack`, which adds a new dimension and needs equal shapes. `cat` joins along an existing one.
  - text: torch.Size([4, 3])
    why: "`cat` doesn't overwrite anything. The 2 new rows are added after the 4 old ones."
hint: "`dim=0` is the row dimension. Count the rows going in."
```

```card
type: fill_blank
id: mean_init
eyebrow: Fill the blanks
title: Start each new row as the mean of its pieces' rows.
code: |
  new_rows = torch.stack([W[ids].mean(dim=[[0]]) for ids in pieces])
  W_ext = torch.cat([[[1]]], dim=0)
tokens: ["0", "1", "[W, new_rows]", "[new_rows, W]"]
answer: ["0", "[W, new_rows]"]
tips:
  "1": "`W[ids]` is `[pieces, 4096]`. `mean(dim=1)` averages the 4096 numbers inside each piece's row and leaves one number per piece. You want the mean across the pieces, down dim 0."
  "[new_rows, W]": That shifts every old row down by the number of new tokens, so every old id reads someone else's row. The old rows must keep their places.
why_ok: "`W[ids]` gathers one row per piece, `mean(dim=0)` averages them into one row, and the new rows go after the old ones so every old id still finds its own row."
why_wrong: "Two checks: average across the pieces, not across the numbers inside a row, and keep the old rows where they were."
hint: "`W[ids]` has one row per piece. Which dimension runs across the pieces?"
```

```card
type: calc
id: new_params
eyebrow: Do the math
title: How many new trainable parameters do 8,000 tokens add to Llama 3.1 8B?
given:
  - { label: New tokens, value: "8,000" }
  - { label: Hidden size, value: "4,096" }
  - { label: Matrices that grow, value: embed_tokens and lm_head }
unit: M params
answer: { min: 65.0, max: 66.0 }
placeholder: "0.0"
why_ok: 8,000 × 4,096 × 2 = 65.5M parameters. With fp32 weights, gradients and Adam's two moments that's 65.5M × 16 bytes ≈ 1.05 GB, which is why 4B.3 trains only the new rows in full and the rest of the model through LoRA.
why_wrong: Each new token gets one row of 4,096 numbers in each of the two matrices.
hint: One row per token per matrix, and every row is 4,096 numbers long.
```

```card
type: quiz
ref: random_rows_hurt
```

```card
type: quiz
ref: append_not_prepend
```

```card
type: quiz
ref: per_word_loss
```

## Build steps

### Grow the matrices

```challenge
id: extend_rows
file: extend.py
minutes: 30
start:
  code: |
    import math
    import re
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # From 4B.1: the corpus, the stand-in for Llama 3.1's vocab, encode(), and your 20 picks.
    CORPUS = """
    Монгол Улс Төв Азид оршдог. Улаанбаатар бол Монгол Улсын нийслэл юм.
    Би өглөө бүр цай уудаг. Манай гэр бүл хотод амьдардаг. Өвөл маш хүйтэн байдаг.
    Хүүхдүүд сургууль руу явж байна. Тэр номын санд ном уншиж байна.
    Малчид зун нүүдэллэдэг. Энэ морь их хурдан гүйдэг. Бид маргааш хөдөө явна.
    Монгол хэл кирилл үсгээр бичигддэг. Наадмын үеэр бөх барилдаж, морь уралдана.
    Ээж маань бууз хийж байна. Тэнгэр цэлмэг, нар хурц байна.
    Оюутнууд их сургуульд суралцаж байна. Миний ах эмчээр ажилладаг.
    Говьд тэмээ олон байдаг. Бид гэртээ хоол идэж байна.
    Хөдөөгийн агаар цэвэр байдаг. Чи хаана амьдардаг вэ?
    Багш хүүхдүүдэд түүх заадаг. Сэлэнгэ мөрөн хойд зүг рүү урсдаг.
    Манай ангийн хүүхдүүд хөл бөмбөг тоглодог. Зун малчид нүүдэг.
    Эмээ маань сүүтэй цай чанаж байна.
    """
    HELD_OUT = """
    Манай хүүхдүүд өглөө бүр сургуульдаа явдаг. Өвөл малчид гэртээ байдаг.
    Тэр их сургуульд түүх заадаг. Бид зун хөдөө амьдардаг.
    Улаанбаатар хотод олон хүн амьдардаг.
    """
    LETTERS = "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
    BASE_VOCAB = set(LETTERS) | set(
        "ал ар ан ах ай ий ин ол он ор от ов ул ур эр ер ен ст та то на но ни ра ро ри ли ла ло "
        "да до ди ба ма ха хо га го де не те ре ле ам ат ад ак эн эл аа ээ уу оо ын ой "
        "сто тор ова ани анд ата ира ора ана ола ага".split())
    NEW_TOKENS = ["хүүхдүүд", "байна", "амьдардаг", "сургууль", "монгол", "байдаг", "малчид",
                  "дэг", "улаанбаатар", "тэ", "нийслэл", "өглөө", "манай", "хүйтэн", "зун",
                  "нүү", "морь", "бид", "хөд", "маань"]


    def words(text):
        return re.findall(r"\w+", text.lower())


    def encode(word, vocab):
        toks, i = [], 0
        while i < len(word):
            j = len(word)
            while j > i + 1 and word[i:j] not in vocab:
                j -= 1
            toks.append(word[i:j])
            i = j
        return toks


    BASE = [" "] + sorted(BASE_VOCAB)  # id 0 is the gap between words
    EXT = BASE + NEW_TOKENS            # new tokens take the next ids, after every old one


    def to_ids(text, tokens):
        """Token ids for text, with id 0 before every word."""
        vocab, index = set(tokens), {t: i for i, t in enumerate(tokens)}
        ids = []
        for w in words(text):
            ids.append(0)
            ids += [index[t] for t in encode(w, vocab)]
        return torch.tensor(ids)


    def pretrain(steps=150):
        """A stand-in for Llama 3.1: a bigram model, an embedding and an LM head with nothing in
        between, trained on the corpus with the base vocab. Takes about a second."""
        ids = to_ids(CORPUS, BASE)
        xs, ys = ids[:-1], ids[1:]
        torch.manual_seed(0)
        model = nn.Sequential(nn.Embedding(len(BASE), 16), nn.Linear(16, len(BASE), bias=False))
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(steps):
            loss = F.cross_entropy(model(xs), ys)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # E is the embedding, [vocab, 16]. H is the LM head's weight, also [vocab, 16].
        return model[0].weight.detach(), model[1].weight.detach()


    E, H = pretrain()

setup:
  - say: "The top of the file is 4B.1 plus a tiny pretrained model, a stand-in for Llama 3.1. E is its embedding, H its LM head. Both have one row per base token."
    pause: 4.2
  - say: "The 20 new tokens get ids after every old one. So each matrix needs 20 more rows, added at the end, and each new row needs a starting value."
    pause: 4.0
  - say: "Here's the stub. Start each new row from the mean of its base pieces' rows. Your turn: fill in extend."
    type_file: extend_rows/stub.py
    pause: 1.2
tests: extend_rows/tests.py
solution: extend_rows/solution.py
alternatives:
  - extend_rows/alternatives/sum_over_len.py
  - extend_rows/alternatives/index_select.py
  - extend_rows/alternatives/loop.py
hints:
  start:
    try: |
      pieces = [BASE.index(p) for p in encode(NEW_TOKENS[1], BASE_VOCAB)]
      print(NEW_TOKENS[1], pieces)
      print(E[pieces].shape)
    see: |
      байна [18, 37, 47]
      torch.Size([3, 16])
    next: "That's the new token `байна` and the three base pieces it splits into today: three rows of 16 numbers each. The new token needs one row of 16 built from those three. Build it for this one token first, then for all 20."
  logic: |
    for each new token in pieces:
      take the rows of weight for its base pieces
        → average them into one row
    stack the averaged rows → new_rows
    old rows first, new_rows after them
      → return the longer matrix
  tools: |
    rows for one token's pieces
      t[ids]                       rows of t picked by a list of ids
    average them into one row
      t.mean(dim)                  average along dimension dim
    one row per new token
      [expr for p in seq]          a list, one item per token
      torch.stack(list)            joins same-shape tensors, new dim 0
    old rows first, new rows after
      torch.cat([a, b], dim)       joins tensors along dimension dim
  nudge: Two steps. Build one row per new token from its pieces, then put all the rows together so the old ones keep their places. Where do the new rows go?
  concept: "`weight[ids]` with a list of piece ids gives one row per piece, shape `[pieces, 16]`. `.mean(dim=0)` averages down the pieces into one `[16]` row. `torch.stack` turns the list of those rows into `[new, 16]`, and `torch.cat([weight, new_rows], dim=0)` adds them after the old rows."
  example: |
    Trace it on a tiny matrix with 3 base rows of width 2:

    - `weight = [[1, 2], [3, 4], [5, 6]]`, `pieces = [[0, 2], [1]]`. Two new tokens.
    - New token 0 splits into base ids 0 and 2: rows `[1, 2]` and `[5, 6]`, mean `[3, 4]`.
    - New token 1 has one piece, id 1: the mean of one row is that row, `[3, 4]`.
    - `new_rows` has shape `[2, 2]`: `[[3, 4], [3, 4]]`.
    - Result: shape `[5, 2]`. Rows 0 to 2 are exactly `weight`, rows 3 and 4 are `new_rows`.

    Check: in the real script, `E` goes from `[102, 16]` to `[122, 16]`.
  skeleton: |
    new_rows = torch.stack([weight[___].mean(dim=_) for ids in ______])
    return torch.cat([______, ________], dim=_)
  solution: |
    new_rows = torch.stack([weight[ids].mean(dim=0) for ids in pieces])
    return torch.cat([weight, new_rows], dim=0)
on_solve: "Both matrices grew from 102 to 122 rows, and \"байна\" starts as the average of ба, й and на. On the GPU, `model.resize_token_embeddings()` adds the rows but doesn't know the pieces: recent transformers versions sample new rows around the mean of all old rows. So you overwrite them with the piece means yourself, as in this step. Next, check that this start doesn't blow up the loss."
mistakes:
  - { file: extend_rows/mistakes/placeholders.py, fails: grows_rows, says: "still returns `...`" }
  - { file: extend_rows/mistakes/rows_placeholder.py, fails: grows_rows, says: "`new_rows` is still `...`" }
  - { file: extend_rows/mistakes/no_cat.py, fails: grows_rows, says: "only the new rows" }
  - { file: extend_rows/mistakes/cat_dim1.py, fails: grows_rows, says: "joins along `dim=1`" }
  - { file: extend_rows/mistakes/mean_dim1.py, fails: grows_rows, says: "`mean(dim=1)`" }
  - { file: extend_rows/mistakes/prepend.py, fails: old_rows_kept, says: "every old id" }
  - { file: extend_rows/mistakes/sum.py, fails: mean_of_pieces, says: "sum of its pieces" }
  - { file: extend_rows/mistakes/first_piece.py, fails: mean_of_pieces, says: "only the first piece" }
  - { file: extend_rows/mistakes/random.py, fails: mean_of_pieces, says: "random" }
  - { file: extend_rows/mistakes/global_mean.py, fails: mean_of_pieces, says: "identical" }
  - { file: extend_rows/mistakes/global_weight.py, fails: mean_of_pieces, says: "ignores the `weight` it's given" }
```

### Check the loss at step 0

```challenge
id: step0_loss
file: extend.py
minutes: 30
start: { from: extend_rows }
setup:
  - say: "Before a GPU spends an hour training these rows, check where they start. Score the held-out text with the extended model: the mean loss of predicting each next token."
    pause: 4.0
  - say: "The model is just E and H now, no nn.Linear. The stub compares four numbers: the base model, a uniform guess, random new rows and your mean-of-pieces rows. Your turn: write step0_loss."
    type_file: step0_loss/stub.py
    pause: 1.2
tests: step0_loss/tests.py
solution: step0_loss/solution.py
alternatives:
  - step0_loss/alternatives/embedding_linear.py
  - step0_loss/alternatives/log_softmax.py
hints:
  start:
    try: |
      ids = to_ids(HELD_OUT, BASE)
      print(len(ids), E[ids[0]].shape, H.shape)
      print((H @ E[ids[0]]).shape)
    see: |
      128 torch.Size([16]) torch.Size([102, 16])
      torch.Size([102])
    next: "That's one position: its embedding row is 16 numbers, and against the 102 rows of the head it gives one score per token, 102 scores. `step0_loss` needs that for every position except the last, then the loss against the token that really comes next."
  logic: |
    drop the last id (nothing follows it)
      → look up its embedding row for each remaining id
      → score every vocab token against each row using H
        (each row of H is one token's vector)  → logits
    targets = the ids from the second one on
      → return the mean cross-entropy of logits vs targets
  tools: |
    inputs and targets
      x[a:b]                       slice of ids
    embedding row per input id
      t[ids]                       rows of t picked by indexes
    score every vocab token against each row
      a @ b                        matrix multiply: [n,d] @ [d,m]
                                   H is [vocab, d]: fit the shapes
    mean loss against the targets
      F.cross_entropy(logits, y)   mean loss from raw scores
  nudge: Every token except the last predicts the one after it. What does `nn.Linear` do with its weight, and how do you write that with `H` alone?
  concept: "`E[ids[:-1]]` looks up a row for every token that has a next one, `[n−1, 16]`. The LM head scores every vocab token against that row: `nn.Linear` computes `x @ weight.T`, so with `H` of shape `[vocab, 16]` the logits are `E[ids[:-1]] @ H.T`, `[n−1, vocab]`. The targets are the tokens one step later, `ids[1:]`, and `F.cross_entropy` takes raw logits."
  example: |
    Trace it on a vocabulary of 3 with width 2: `E = [[1, 0], [0, 1], [1, 1]]`, `H = [[1, 0], [0, 1], [0, 0]]`, `ids = [0, 1, 1]`.

    - Inputs: the first two ids, `[0, 1]`. Their rows of `E` are `[[1, 0], [0, 1]]`, shape `[2, 2]`.
    - Logits: row 0 is `[1, 0, 0]`, row 1 is `[0, 1, 0]`. Shape `[2, 3]`, one score per vocab token.
    - Targets: `[1, 1]`.
    - Position 0 gives the target a score of 0 against a sum of e + 2 over everything: loss ln(e + 2) ≈ `1.551`.
    - Position 1 gives the target a score of 1: loss ln(e + 2) − 1 ≈ `0.551`.
    - Mean: `1.051`.

    Check: a uniform guess over 3 tokens is ln 3 ≈ 1.099, so this model is slightly better than guessing.
  skeleton: |
    logits = E[ids[___]] @ _.__
    return F.cross_entropy(______, ids[__])
  solution: |
    logits = E[ids[:-1]] @ H.T
    return F.cross_entropy(logits, ids[1:])
on_solve: "Random rows start worse than a uniform guess: a few new tokens get huge logits everywhere and soak up probability. Mean-of-pieces starts lowest. Its per-token loss is above the base model's, but each extended token carries more text. Multiply each loss by its token count to compare totals: the extended model is already ahead. Try `torch.zeros` for the new rows, and `E.mean(dim=0)` for all of them, then see where they land."
mistakes:
  - { file: step0_loss/mistakes/placeholders.py, fails: returns_loss, says: "still returns `...`" }
  - { file: step0_loss/mistakes/no_transpose.py, fails: returns_loss, says: "`H.T`" }
  - { file: step0_loss/mistakes/no_shift.py, fails: next_token, says: "predicts each token from itself" }
  - { file: step0_loss/mistakes/backwards.py, fails: next_token, says: "the token before" }
  - { file: step0_loss/mistakes/softmax_first.py, fails: next_token, says: "already applies log-softmax" }
  - { file: step0_loss/mistakes/sum.py, fails: next_token, says: "sum" }
  - { file: step0_loss/mistakes/tied.py, fails: next_token, says: "`E.T`" }
  - { file: step0_loss/mistakes/no_main.py, fails: script_runs, says: "is missing" }
```

### Extend Llama 3.1 8B

```replay
id: extend_llama
intro: "On the GPU the same two steps take a few minutes: resize both matrices, fill the new rows from their Llama pieces, and score 4A.3's held-out split before any training. Watch a recording first."
log: replays/extend-llama.jsonl
speed: 40
recorded:
  hardware: RTX 3060 12 GB
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: Step-0 loss by initialisation (nats per token)
  flag_above: 11.82
  x_labels: ["random N(0, 1)", mean of pieces]
  caption: The red bar is above ln(136,256) ≈ 11.82, a uniform guess over the extended vocab. Mean of pieces starts at 3.83 nats per token, 1.08 bits per character against Llama's 1.04.
summary:
  - { label: Vocab, value: "128,256 → 136,256" }
  - { label: New parameters, value: 65.5M (embed + head) }
  - { label: Step-0 loss, value: 3.83 nats per token (1.08 bits/char) }
run_yourself: |
  # on your GPU, from the course repo
  python extend_vocab.py \
    --base meta-llama/Llama-3.1-8B --tokenizer tokenizers/llama31-mn-8k \
    --init mean-of-pieces --eval data/mn-clean/heldout.jsonl \
    --out runs/llama31-mn-8k-init
```

### Record your metrics

```metrics
id: record
intro: Your step-0 loss with mean-of-pieces rows, and the size of the extended vocab. They're compared with the reference run and the pass limit.
```

## Hints

- `model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)` grows both matrices and keeps the old rows. Then overwrite the new rows under `torch.no_grad()`: `emb.weight[i] = emb.weight[pieces].mean(0)`, and the same for `model.lm_head.weight`.
- Get each new token's pieces from the **base** tokenizer: `base_tok(token_text, add_special_tokens=False).input_ids`. A word-initial token and the same text mid-word can split differently, so encode it the way it appears in text (with Llama's leading space where it has one).
- Check tied weights first: `model.config.tie_word_embeddings` is `False` for Llama 3.1 8B, so both matrices need new rows. Smaller Llama 3.2 models tie them, and one resize covers both.
- If the step-0 loss is above ln(vocab size), stop. Something is misaligned (rows prepended, wrong pieces, or an fp16 overflow in the new rows).
- Save the resized `embed_tokens` and `lm_head` in bf16 (2.2 GB) with the list of new ids. The rest of Llama is unchanged, and 4B.3 loads it in 4-bit as the control did.

## Done when

- [ ] Both matrices grow, with each new row the mean of its pieces {#rows_grown auto=challenge:extend_rows}
- [ ] Step-0 loss checked against random rows and a uniform guess {#step0_checked auto=challenge:step0_loss}
- [ ] Llama 3.1 8B resized and initialised, step-0 loss under 5 nats per token on your held-out split {#llama_extended}
- [ ] Step-0 loss and vocab size recorded {#metrics_recorded auto=metrics}
