---
id: "4B.1"
title: Train a Mongolian tokenizer
summary: Measure how many tokens Llama 3.1 spends on each Mongolian word, train a Mongolian BPE, and choose the new tokens worth adding.
phase: "4B"
hours: 8
type: research
status: ready
protected: false
cut: null
prerequisites:
  - 4A.5
  - "1.6"
papers:
  - title: Neural Machine Translation of Rare Words with Subword Units
    authors: Sennrich et al.
    year: 2016
    url: https://arxiv.org/abs/1508.07909
quality_metric:
  id: fertility
  label: Tokens per Mongolian word, new tokenizer
  unit: tok/word
  note: Measured on held-out sentences from your 4A.3 corpus with Llama 3.1's tokenizer plus your new tokens. Your 1.6 audit gave Llama 3.1 alone (reference ≈3.4).
  inputs:
    - { id: fertility, label: "Held-out tokens per word, extended tokenizer", hint: Total tokens / whitespace words on held-out text }
  compute: value
  reference: 1.62
  pass: { max: 2.0 }
systems_metric:
  id: new_tokens
  label: Tokens added
  unit: count
  note: Every new token is one new embedding row and one new LM-head row to train in 4B.2 and 4B.3.
  inputs:
    - { id: new_tokens, label: Tokens added to Llama 3.1's vocab, hint: Length of your new_tokens.txt }
  compute: value
  reference: 8000
  pass: { max: 16000 }
quiz:
  - id: held_out
    q: You measure fertility on the same sentences you trained the Mongolian BPE on. What goes wrong?
    answer: 1
    options:
      - text: Nothing. Text is text, so the number is the same on any Mongolian sentences
        why: It isn't. A trainer that saw a sentence has merges for its exact words, so those words come out as one token each. New text has words it never saw.
      - text: Fertility looks better than it is, because the trainer memorised the training words
        why: Right. On a small corpus BPE learns whole training words as single tokens. Only held-out text shows how it does on words it hasn't seen, which is what your users will type.
      - text: Fertility looks worse, because the training text is harder
        why: The training text is the easiest text there is for this tokenizer. Every frequent word in it got its own merges.
  - id: how_many
    q: Your 16k Mongolian BPE has 13,086 tokens that Llama 3.1 lacks. Why add only the top 8,000?
    answer: 2
    options:
      - text: Llama 3.1 can't hold more than 136,256 tokens
        why: Nothing caps the vocab there. You can resize the embedding to any size. The limits are training data and memory, not a hard maximum.
      - text: The other 5,086 are copies of tokens Llama already has
        why: The candidates already exclude Llama's own tokens. The low-ranked ones are new, they just save very little.
      - text: Each new token is a row to train from scratch, and the low-ranked ones are rare and save little
        why: Right. Token 12,000 might appear a few hundred times in the corpus's 62M words, too few updates to learn a good embedding, and fertility has nearly stopped improving by then.
  - id: rank_by_savings
    q: "\"ын\" appears 80,000 times and Llama splits it into 2 tokens. \"байгаа\" appears 25,000 times and Llama splits it into 5. Which saves more tokens as a new token?"
    answer: 1
    options:
      - text: "\"ын\", because it's more than 3× as frequent"
        why: "Each \"ын\" saves only 1 token (2 pieces become 1), so 80,000 × 1 = 80,000 tokens saved."
      - text: "\"байгаа\", 100,000 tokens saved against 80,000"
        why: "Each \"байгаа\" turns 5 pieces into 1 and saves 4: 25,000 × 4 = 100,000. Frequency alone ranks them the wrong way round."
      - text: They save the same, because both become one token
        why: Both become one token, but they started from different numbers of pieces. What a token saves is the pieces it replaces minus the one it costs.
---

## Goal

Llama 3.1 was trained mostly on English. Your 1.6 audit measured how many tokens it spends on a Mongolian word: about **3.4** in the reference run, against 1.3 for English. So a Mongolian answer costs about 2.6× what the same answer costs in English, and a context window holds 2.6× fewer words. In this lesson you measure **fertility** (tokens per word) yourself, train a Mongolian BPE on a small Cyrillic corpus, and choose which of its tokens to add to Llama's vocab: the ones that save the most tokens across the corpus. Then you do it for real on your 4A.3 corpus.

## Learn

```card
type: concept
id: why_fertility
eyebrow: Why a new tokenizer
title: The same sentence costs Llama 3.1 more than twice as many tokens in Mongolian.
visual:
  kind: bars
  max: 3.5
  bars:
    - { label: English, value: 1.31, text: 1.31 tok/word }
    - { label: Mongolian, value: 3.38, text: 3.38 tok/word }
    - { label: + 8k tokens, value: 1.62, text: 1.62 tok/word }
body: |
  **Fertility** is tokens per word: total tokens divided by whitespace words. Llama 3.1's 128,256 tokens include plenty of English and Russian pieces but few Mongolian ones, so a word like **хүүхдүүд** (children) falls apart into four or five pieces. More tokens per word means slower generation, a shorter effective context and more steps for the model to get a word right. Adding 8,000 Mongolian tokens brings it to about 1.6.
```

```card
type: predict
id: predict_encode
eyebrow: Predict the output
title: Greedy longest match takes the longest piece in the vocab at each position. What prints?
code: |
  vocab = {"б", "а", "й", "н", "ай", "на", "бай"}
  print(encode("байна", vocab))
options:
  - text: "['б', 'ай', 'на']"
    why: That starts with the shortest piece. At position 0 the longest piece in the vocab that fits is "бай", three letters, so it wins.
  - text: "['бай', 'на']"
    correct: true
    why: "At position 0 \"байна\" and \"байн\" aren't in the vocab, \"бай\" is, so it's taken. At position 3 the longest match is \"на\". 2 tokens for 1 word."
  - text: "['байна']"
    why: "\"байна\" isn't in this vocab. Adding it is exactly what this lesson is about, and then it would be one token."
  - text: "['б', 'а', 'й', 'н', 'а']"
    why: That's what you get with only single letters. Every multi-letter piece in the vocab that fits is taken first.
hint: Start at the first letter and try the longest string first, "байна", then "байн", then "бай". Take the first one that's in the vocab.
```

```card
type: calc
id: context_words
eyebrow: Do the math
title: How many Mongolian words fit in an 8,192-token context with Llama 3.1's tokenizer?
given:
  - { label: Context, value: "8,192 tokens" }
  - { label: Fertility, value: 3.38 tokens per word (1.6's reference) }
unit: words
answer: { min: 2380, max: 2470 }
placeholder: "0"
why_ok: 8,192 / 3.38 ≈ 2,420 words. At 1.62 tokens per word the same window holds about 5,060 words, more than twice as much.
why_wrong: Fertility is tokens per word, so the number of words is the token budget divided by it.
hint: Each word costs 3.38 tokens. Divide the tokens you have by the cost of one word.
```

```card
type: fill_blank
id: savings_score
eyebrow: Fill the blanks
title: Score each candidate by the tokens it saves.
code: |
  saved = {t: counts[t] * (len(encode(t, [[0]])) - [[1]])
           for t in candidates}
tokens: [base_vocab, mn_vocab, "1", "0"]
answer: [base_vocab, "1"]
tips:
  mn_vocab: Every candidate is already one token in the Mongolian vocab, so this would score 1 − 1 = 0 for all of them. What matters is how many pieces Llama's vocab spends on it today.
  "0": The new token still costs one token itself. A word Llama splits into 2 pieces saves 1 token per use, not 2.
why_ok: Each use of `t` replaces the pieces the base vocab needs with one token. Multiply that saving by how often `t` appears and you have the tokens saved across the corpus.
why_wrong: "Two things to get right: which vocab decides how many pieces `t` costs today, and what the new token itself costs."
hint: Ask what changes when `t` becomes a token. Before, Llama spent some number of pieces on it. After, it spends one.
```

```card
type: reorder
id: tokenizer_pipeline
eyebrow: Put it in order
title: Arrange the research loop for a new tokenizer.
lines:
  - Train a Mongolian BPE on the training split of the corpus
  - Count how often each of its tokens appears in the corpus
  - Score the tokens Llama lacks by base tokens saved
  - Add the top k to Llama's vocab
  - Measure fertility on held-out sentences
shuffle: [3, 1, 4, 0, 2]
why_ok: Train, count, score, add, measure. The last step uses held-out text, so the number says how the tokenizer does on words it hasn't seen.
why_wrong: "Each step needs the one before it: you can't count tokens before the BPE exists, or score them before you've counted. Measuring comes last, on text the trainer never saw."
hint: Start with the step that needs nothing but the corpus, and end with the step that checks the result.
```

```card
type: quiz
ref: held_out
```

```card
type: quiz
ref: how_many
```

```card
type: quiz
ref: rank_by_savings
```

## Build steps

### Measure fertility

```challenge
id: fertility
file: tokenizer.py
minutes: 20
start:
  code: |
    import re

    # A small Mongolian corpus to train on. Your real one is the 4A.3 corpus, about 62M words.
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
    # Sentences the trainer never sees.
    HELD_OUT = """
    Манай хүүхдүүд өглөө бүр сургуульдаа явдаг. Өвөл малчид гэртээ байдаг.
    Тэр их сургуульд түүх заадаг. Бид зун хөдөө амьдардаг.
    Улаанбаатар хотод олон хүн амьдардаг.
    """

    LETTERS = "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
    # A stand-in for Llama 3.1's vocab as far as Mongolian goes: every letter, plus short
    # pieces it mostly learned from Russian.
    BASE_VOCAB = set(LETTERS) | set(
        "ал ар ан ах ай ий ин ол он ор от ов ул ур эр ер ен ст та то на но ни ра ро ри ли ла ло "
        "да до ди ба ма ха хо га го де не те ре ле ам ат ад ак эн эл аа ээ уу оо ын ой "
        "сто тор ова ани анд ата ира ора ана ола ага".split())


    def words(text):
        """Lowercased words, punctuation dropped. Lowercasing stretches a tiny corpus further."""
        return re.findall(r"\w+", text.lower())


    def encode(word, vocab):
        """Greedy longest match: at each position, take the longest piece that's in vocab.
        A stand-in for BPE's merge order that lets you add tokens by adding strings."""
        toks, i = [], 0
        while i < len(word):
            j = len(word)
            while j > i + 1 and word[i:j] not in vocab:
                j -= 1
            toks.append(word[i:j])
            i = j
        return toks


    def train_bpe(text, n_merges):
        """Your BPE trainer from 1.5, condensed. Returns the vocab: letters plus one token per merge."""
        counts = {}
        for w in words(text):
            counts[w] = counts.get(w, 0) + 1
        corpus = {tuple(w): c for w, c in counts.items()}
        vocab = {ch for w in corpus for ch in w}
        for _ in range(n_merges):
            pairs = {}
            for w, c in corpus.items():
                for a, b in zip(w, w[1:]):
                    pairs[a, b] = pairs.get((a, b), 0) + c
            if not pairs:
                break
            a, b = max(pairs, key=pairs.get)
            vocab.add(a + b)
            merged = {}
            for w, c in corpus.items():
                out, i = [], 0
                while i < len(w):
                    if i + 1 < len(w) and (w[i], w[i + 1]) == (a, b):
                        out.append(a + b)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                merged[tuple(out)] = c
            corpus = merged
        return vocab

setup:
  - say: "Everything you need is already in the file: a small Mongolian corpus, held-out sentences it never sees, a stand-in for Llama 3.1's vocab, words(), encode() and your BPE trainer from 1.5."
    pause: 4.5
  - say: "Fertility is one number, tokens per word. Split the text into words, encode each one, and count."
    pause: 3.2
  - say: "Here's the stub. It trains a Mongolian BPE with 120 merges and compares it with the base vocab on the held-out text. Your turn: fill in fertility."
    type_file: fertility/stub.py
    pause: 1.2
tests: fertility/tests.py
solution: fertility/solution.py
alternatives:
  - fertility/alternatives/loop.py
  - fertility/alternatives/one_line.py
hints:
  logic: |
    split text into words → ws
    for each word in ws:
      encode it with vocab → count the tokens it makes
    add the counts → n_tokens
      → return n_tokens ÷ number of words
  tools: |
    split text into words
      words(text)                  the helper in the file
    tokens per word
      encode(w, vocab)             list of tokens for one word
      len(x)                       how many items are in x
    add up over all words
      sum(expr for w in seq)       adds one value per word
    tokens per word, overall
      plain arithmetic: /
  nudge: Three numbers in a row. The words, the tokens all those words turn into, and one divided by the other. Which goes on top?
  concept: "`words(text)` gives the list of words, split the same way the corpus is. `encode(w, vocab)` gives the list of tokens for one word, so its `len` is that word's token count. Add those up over every word, then divide by the number of words. Use the `vocab` you're given, so the same function measures any tokenizer."
  example: |
    Trace it on `text = "Цай, байна."` with `BASE_VOCAB`:

    - `words(text)` lowercases and drops punctuation: `["цай", "байна"]`. Two words.
    - `encode("цай", BASE_VOCAB)` is `["ц", "ай"]`, 2 tokens.
    - `encode("байна", BASE_VOCAB)` is `["ба", "й", "на"]`, 3 tokens.
    - `n_tokens` = 2 + 3 = 5. Fertility = 5 / 2 = `2.5`.

    Now add `"байна"` to the vocab. It encodes as a single piece, so `n_tokens` = 2 + 1 = 3 and fertility drops to `1.5`. Same function, different `vocab`.
  skeleton: |
    ws = _____(text)
    n_tokens = sum(len(______(w, _____)) for w in ws)
    return ________ / len(__)
  solution: |
    ws = words(text)
    n_tokens = sum(len(encode(w, vocab)) for w in ws)
    return n_tokens / len(ws)
on_solve: "That's the gap 4B is about: the base vocab spends over 4 tokens per word on text it has never seen, the Mongolian BPE under 2. Two caveats before you trust the number. The base vocab here is a toy, so the real gap (about 3.4 vs 1.6) is smaller. And 120 merges on a tiny corpus learn whole words, which helps on held-out text only where the words repeat. Try 40 and 200 merges in `train_bpe`: held-out fertility goes from 3.67 to 1.46. On this tiny held-out set more merges keep helping because its words repeat the corpus's. On real text the curve flattens, as the replay at the end shows."
mistakes:
  - { file: fertility/mistakes/placeholders.py, fails: returns_number, says: "still returns `...`" }
  - { file: fertility/mistakes/ws_placeholder.py, fails: returns_number, says: "`ws` is still `...`" }
  - { file: fertility/mistakes/tuple.py, fails: returns_number, says: "returns 2 values" }
  - { file: fertility/mistakes/letters.py, fails: tokens_per_word, says: "you counted letters" }
  - { file: fertility/mistakes/inverted.py, fails: tokens_per_word, says: "words per token" }
  - { file: fertility/mistakes/unique.py, fails: tokens_per_word, says: "distinct tokens" }
  - { file: fertility/mistakes/whole_text.py, fails: tokens_per_word, says: "spaces and newlines became tokens" }
  - { file: fertility/mistakes/split.py, fails: tokens_per_word, says: "`text.split()`" }
  - { file: fertility/mistakes/global_vocab.py, fails: tokens_per_word, says: "ignores the `vocab` it's given" }
  - { file: fertility/mistakes/no_main.py, fails: script_runs, says: "are missing" }
```

### Pick the tokens to add

```challenge
id: pick_new_tokens
file: tokenizer.py
minutes: 30
start: { from: fertility }
setup:
  - say: "Fertility works. The Mongolian BPE has 150 tokens, but you won't add them all to Llama. Each one is a new row to train. First, count how often each Mongolian token shows up in the corpus."
    type: |

      def token_counts(text, vocab):
          """How often each token appears when every word of text is encoded with vocab."""
          counts = {}
          for w in words(text):
              for t in encode(w, vocab):
                  counts[t] = counts.get(t, 0) + 1
          return counts

      COUNTS = token_counts(CORPUS, MN_VOCAB)
    pause: 2.6
  - say: "A token is worth adding when it saves a lot of base tokens. That depends on how often it appears and on how many pieces the base vocab spends on it today."
    pause: 4.0
  - say: "Here's the stub. Your turn: rank the candidates by tokens saved and return the top k."
    type_file: pick_new_tokens/stub.py
    pause: 1.2
tests: pick_new_tokens/tests.py
solution: pick_new_tokens/solution.py
alternatives:
  - pick_new_tokens/alternatives/sort_items.py
  - pick_new_tokens/alternatives/tie_break.py
  - pick_new_tokens/alternatives/no_filter.py
hints:
  logic: |
    for each token t in counts:
      already in base_vocab?
        yes → skip it
        no  → it's a candidate
    for each candidate:
      pieces = how many tokens base_vocab splits it into
      saved = counts[t] × (pieces − 1)   (it costs 1 once added)
    sort the candidates by saved, biggest first
      → return the first k
  tools: |
    keep tokens the base vocab lacks
      x not in coll                membership test
      [t for t in d if cond]       list filtered while it is built
    base pieces per candidate
      encode(t, vocab)             list of tokens for t
      len(x)                       how many items are in x
    saved per candidate
      {k: expr for k in seq}       dict, one entry per key
      plain arithmetic: * -        uses × (pieces − 1)
    best k first
      sorted(seq, key=fn, reverse=flag)   a new sorted list
      d.get                        looks up a key in dict d
      seq[:n]                      the first n items
  nudge: Three steps. Throw out what the base already has, give everything left a score, keep the best k. What should the score measure?
  concept: "A candidate `t` that the base vocab encodes as `len(encode(t, base_vocab))` pieces costs 1 token once it's added, so each use saves that many minus 1. Multiply by `counts[t]`, how often it appears. Then `sorted(saved, key=saved.get, reverse=True)` lists the tokens from the biggest saver down, and `[:k]` keeps the first k."
  example: |
    Take `counts = {"байна": 3, "ай": 4, "сургууль": 1, "цай": 2}` and `base_vocab = BASE_VOCAB`.

    - `"ай"` is already in the base vocab, so it is dropped. That's the most frequent token, but it saves nothing.
    - `"байна"` encodes as 3 pieces: saves 2 per use × 3 uses = `6`.
    - `"сургууль"` encodes as 6 pieces: saves 5 per use × 1 use = `5`.
    - `"цай"` encodes as 2 pieces: saves 1 per use × 2 uses = `2`.
    - Ranked: `["байна", "сургууль", "цай"]`. With `k = 2` you return `["байна", "сургууль"]`.

    Check: sorting by `counts` alone would put `"ай"` first, which is why the saving matters.
  skeleton: |
    candidates = [t for t in counts if t not in __________]
    saved = {t: counts[t] * (len(encode(t, __________)) - _) for t in candidates}
    return sorted(saved, key=_________, reverse=____)[:_]
  solution: |
    candidates = [t for t in counts if t not in base_vocab]
    saved = {t: counts[t] * (len(encode(t, base_vocab)) - 1) for t in candidates}
    return sorted(saved, key=saved.get, reverse=True)[:k]
on_solve: "Look at the curve. The first 5 tokens cut more than a whole token per word, the next 15 together about as much again, because the biggest savers are frequent words the base vocab shreds, like байна and сургууль. That flattening is why the real run stops at 8,000 of 13,086 candidates. Two things to try: pick by raw count instead (`saved = counts`) and watch the base vocab's own letters crowd the top."
mistakes:
  - { file: pick_new_tokens/mistakes/placeholders.py, fails: returns_tokens, says: "still returns `...`" }
  - { file: pick_new_tokens/mistakes/saved_placeholder.py, fails: returns_tokens, says: "`saved` is still `...`" }
  - { file: pick_new_tokens/mistakes/returns_dict.py, fails: returns_tokens, says: "returns a dict" }
  - { file: pick_new_tokens/mistakes/pairs.py, fails: returns_tokens, says: "(token, score) pairs" }
  - { file: pick_new_tokens/mistakes/no_cut.py, fails: returns_tokens, says: "asked for 3" }
  - { file: pick_new_tokens/mistakes/raw_count.py, fails: best_savers, says: "how often they appear" }
  - { file: pick_new_tokens/mistakes/raw_count_unfiltered.py, fails: best_savers, says: "already in the base vocab" }
  - { file: pick_new_tokens/mistakes/letters_saved.py, fails: best_savers, says: "letters" }
  - { file: pick_new_tokens/mistakes/no_minus_one.py, fails: best_savers, says: "costs one token itself" }
  - { file: pick_new_tokens/mistakes/ascending.py, fails: best_savers, says: "smallest savers first" }
  - { file: pick_new_tokens/mistakes/global_base.py, fails: best_savers, says: "ignores the `base_vocab` it's given" }
```

### Train the real tokenizer

```replay
id: train_tokenizer
intro: "On your 4A.3 corpus this is a CPU job: about 10 minutes to train a 16k Mongolian BPE, then the same ranking you just wrote over its tokens. Watch a recording first."
log: replays/train-tokenizer.jsonl
speed: 40
recorded:
  hardware: 16-core CPU
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: Held-out tokens per word as tokens are added
  flag_above: 2.0
  x_labels: [Llama 3.1, "+ 13,086"]
  caption: Red bars are above 2.0 tokens per word. The first 2,000 tokens do most of the work. After 8,000 each extra thousand saves less than 1%, while every one still adds two rows to train.
summary:
  - { label: Held-out fertility, value: 3.38 → 1.62 tok/word }
  - { label: Tokens added, value: "8,000" }
  - { label: Extended vocab, value: "128,256 → 136,256" }
run_yourself: |
  # on your machine, from the course repo (CPU is fine)
  python train_tokenizer.py \
    --corpus data/mn-clean/train.jsonl --heldout data/mn-clean/heldout.jsonl \
    --base meta-llama/Llama-3.1-8B --vocab-size 16000 --add 8000 \
    --out tokenizers/llama31-mn-8k
```

### Record your metrics

```metrics
id: record
intro: Held-out fertility with your extended tokenizer, and how many tokens you added. They're compared with the reference run and the pass limit.
```

## Hints

- Split train and held-out by document, not by sentence. Near-duplicate news articles across the split make held-out fertility look better than it is.
- Train the Mongolian BPE with the same pre-tokenizer as Llama 3.1 (its regex split and byte-level alphabet), or its tokens won't line up with how Llama cuts text. `tokenizers`' `BpeTrainer` on a copy of Llama's pre-tokenizer is the simplest way.
- `tokenizer.add_tokens()` matches added tokens before BPE runs, which is fine for whole words but slow for thousands of them. Merging the new tokens into the BPE model itself (vocab and merges, as Chinese LLaMA does with SentencePiece) keeps encoding fast.
- Make the extended vocab a multiple of 64 (128,256 + 8,000 = 136,256 is). Matmuls over the LM head run faster on aligned sizes.
- Keep `new_tokens.txt` in rank order. 4B.2 needs each new token's base pieces, and 4B.4 needs to know which tokens are new.

## Done when

- [ ] Fertility measured on held-out text for the base and Mongolian vocabs {#fertility_measured auto=challenge:fertility}
- [ ] New tokens ranked by base tokens saved {#tokens_ranked auto=challenge:pick_new_tokens}
- [ ] Extended tokenizer saved, under 2.0 tokens per word on your 4A.3 held-out split {#tokenizer_saved}
- [ ] Fertility and tokens added recorded {#metrics_recorded auto=metrics}
