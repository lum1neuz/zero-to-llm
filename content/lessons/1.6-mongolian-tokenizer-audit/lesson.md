---
id: "1.6"
title: Mongolian tokenizer audit
summary: Measure how a tokenizer treats Mongolian Cyrillic, from fertility to byte fallback to vocab coverage, then run the same audit on Llama 3.1.
phase: "1"
hours: 4
type: research
status: ready
protected: false
cut: null
prerequisites:
  - "1.5"
papers: []
quality_metric:
  id: fertility
  label: Tokens per Mongolian word (Llama 3.1 tokenizer)
  unit: tok/word
  note: From your own audit run. The reference is the illustrative recording, not a measured value. Above 6 usually means the words were counted wrong.
  inputs:
    - { id: fertility, label: Tokens per Mongolian word, hint: "audit_tokenizer.py prints it under \"fertility\"" }
  compute: value
  reference: 3.38
  pass: { max: 6.0 }
systems_metric:
  id: cyrillic_coverage
  label: Share of Cyrillic merges in the vocab
  unit: "%"
  note: Your cyrillic_share over Llama 3.1's 127,744 merged tokens. The reference is the illustrative recording, not a measured value.
  inputs:
    - { id: cyrillic_coverage, label: Merged tokens with Cyrillic (%), hint: "audit_tokenizer.py prints it, or run your cyrillic_share on the vocab" }
  compute: value
  reference: 4.6
  pass: { min: 0.5 }
quiz:
  - id: token_tax
    q: A tokenizer needs 2.6× as many tokens for Mongolian as for the same text in English. What does that cost a Mongolian user?
    answer: 1
    options:
      - text: Nothing; tokens are just an internal detail
        why: Everything in an LLM is counted in tokens. Context windows, compute, latency and API prices all scale with them.
      - text: 2.6× the context, compute and price for the same text, and 2.6× slower generation per word
        why: Every token costs a forward pass. Mongolian fills the context 2.6× faster, and every generated word takes 2.6× as many steps.
      - text: Lower quality, but the same speed and cost
        why: The cost is direct. Quality usually suffers too, but more tokens per word costs more compute whatever the quality.
      - text: Mongolian can't be generated at all
        why: Byte-level BPE can produce any byte sequence, so any Mongolian text can be generated. It just takes more steps.
  - id: unseen_word
    q: Llama 3.1 has no token for a rare Mongolian word. How does it encode it?
    answer: 0
    options:
      - text: As smaller pieces it does have, down to single bytes if it must
        why: Byte-level BPE always has the 256 bytes to fall back on. A rare word becomes several common pieces, and a letter with no merge becomes its two UTF-8 bytes.
      - text: As one <unk> token
        why: Byte-level tokenizers have no unknown token. Every text is bytes, and every byte has an id.
      - text: It skips the word
        why: Nothing is dropped. decode(encode(text)) always gives the text back, as in 1.5.
      - text: It transliterates it to Latin letters first
        why: The tokenizer never changes the text. It only splits its bytes.
  - id: new_tokens
    q: Lesson 4B.2 adds Mongolian tokens to Llama's vocab. Why can't you add them to the tokenizer and leave the model as it is?
    answer: 2
    options:
      - text: The vocab size has to stay a power of two
        why: Llama 3.1's vocab is 128,256, which isn't a power of two. Sizes are often rounded to a multiple of 64 for speed, but anything works.
      - text: Tokenizer files can't be edited
        why: They're plain JSON. Editing them is the easy part.
      - text: The model has never seen the new ids, so their embedding rows and output weights are untrained
        why: A new id indexes a row of the embedding table and a row of lm_head that don't exist yet. They need a sensible start (4B.2) and training (4B.3) before the model can use them.
      - text: New tokens would change how English is encoded
        why: Merges added after the existing ones only fire on byte pairs they match. English text never contains Cyrillic bytes, so it encodes exactly as before.
---

## Goal

Audit how a tokenizer treats **Mongolian Cyrillic**. Write three measures: **fertility** (tokens per word), the share of tokens that are **byte pieces** of a letter, and the share of a vocabulary's merges that contain **Cyrillic** at all. Try them on two small tokenizers in the browser, one that has never seen Mongolian and one trained on a few paragraphs of it. Then run the same audit on **Llama 3.1's** tokenizer, the one phase 4 fine-tunes.

## Learn

```card
type: concept
id: no_merges_no_words
eyebrow: The problem
title: A tokenizer that never saw Mongolian spells it out byte by byte.
visual:
  kind: bars
  max: 12
  bars:
    - { label: bytes only, value: 11.52, text: "11.5" }
    - { label: English BPE, value: 11.52, text: "11.5" }
    - { label: Mongolian BPE, value: 5.96, text: "6.0" }
body: |
  Every Cyrillic letter is 2 bytes in UTF-8. A byte-level tokenizer with no merges for those bytes spends one token per byte, two per letter. The bars are tokens per word on six Mongolian sentences, measured in this lesson's browser code: 1.5's tokenizer trained on an English story does no better than raw bytes, and 40 merges learned from a few Mongolian paragraphs halve the count. **Fertility**, tokens per word, is the number to watch.
```

```card
type: predict
id: predict_letter_bytes
eyebrow: Predict the output
title: What does this print?
code: |
  print(list("ө".encode("utf-8")))
options:
  - text: "[211, 169]"
    correct: true
    why: ө is U+04E9. UTF-8 stores it in two bytes, 0xD3 0xA9. A tokenizer with no merge for that pair spends two tokens on one letter.
  - text: "[1257]"
    why: 1257 is ө's code point, `ord("ө")`. encode gives the UTF-8 bytes, and no byte is over 255.
  - text: "[169]"
    why: That's only the second byte. Every Cyrillic letter takes two.
  - text: "[208, 169]"
    why: 0xD0 starts the basic Cyrillic letters like а and б. ө sits later in the Cyrillic block, so its first byte is 0xD3.
hint: Cyrillic letters take two bytes each in UTF-8, and each byte is a number from 0 to 255.
```

```card
type: fill_blank
id: fertility_formula
eyebrow: Fill the blanks
title: Define fertility over a sample.
code: |
  fertility = [[0]] / [[1]]
tokens: [total tokens, total words, total characters, number of sentences]
answer: [total tokens, total words]
tips:
  total characters: Tokens per character hides the word structure, and characters differ in bytes between scripts. Fertility is per word, so languages compare fairly.
  number of sentences: Sentences vary in length. Per word is what makes two samples comparable.
why_ok: "Total tokens over total words, summed across the whole sample, so long sentences count for more. English with a good tokenizer sits around 1.3."
why_wrong: "Fertility asks how many tokens one word costs on average."
hint: The name says it. How many tokens does one word cost?
```

```card
type: calc
id: words_in_context
eyebrow: Do the math
title: How many Mongolian words fit in an 8,192-token context at 3.4 tokens per word?
given:
  - { label: Context, value: "8,192 tokens" }
  - { label: Mongolian fertility, value: 3.4 tokens per word }
  - { label: English fertility, value: 1.3 tokens per word }
unit: words
answer: { min: 2380, max: 2440 }
placeholder: "0"
why_ok: 8,192 / 3.4 ≈ 2,410 words. The same window holds about 6,300 English words, 2.6× as many.
why_wrong: Divide the context by the tokens each word costs.
hint: Each Mongolian word uses 3.4 of the 8,192 tokens.
```

```card
type: quiz
ref: token_tax
```

```card
type: quiz
ref: unseen_word
```

```card
type: quiz
ref: new_tokens
```

## Build steps

### Measure fertility

```challenge
id: fertility
file: audit.py
minutes: 20
start:
  code: |
    # 1.5's byte-level BPE, ready-made: train(text, vocab_size) -> merges, vocab; encode(text, merges) -> ids
    def get_stats(ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(ids, pair, idx):
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(idx)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(text, vocab_size):
        ids = list(text.encode("utf-8"))
        merges, vocab = {}, {i: bytes([i]) for i in range(256)}
        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            ids = merge(ids, pair, 256 + i)
            merges[pair] = 256 + i
            vocab[256 + i] = vocab[pair[0]] + vocab[pair[1]]
        return merges, vocab

    def encode(text, merges):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            ids = merge(ids, pair, merges[pair])
        return ids

    english = """Once upon a time, there was a little bird named Tumen. Tumen lived in a tall tree by a quiet river. Every morning, Tumen sang a happy song for the other birds in the tree.
    One day, the wind was very strong. The wind blew the nest out of the tree and into the river. Tumen was sad. He did not know what to do.
    A kind old turtle saw the nest in the water. The turtle swam to the nest and pushed it to the shore. From that day on, the little bird and the old turtle were best friends."""

    mongolian_train = """Монгол Улс бол Төв Азид оршдог улс юм. Монгол Улсын нийслэл нь Улаанбаатар хот.
    Монголчууд эрт дээр үеэс мал аж ахуй эрхэлж ирсэн. Нүүдэлчид гэр бүлээрээ малаа хариулж, улирал бүр нүүдэг.
    Монгол гэр нь эсгий, мод, олсоор хийгдсэн байдаг. Говь цөлд тэмээ олон байдаг.
    Өвөл маш хүйтэн, зун богино боловч сайхан. Хүүхдүүд намар сургуульдаа буцаж ирдэг.
    Наадмын үеэр бөх, морин уралдаан, сур харваа болдог. Морь бол монголчуудын хамгийн үнэнч нөхөр.
    Би өдөр бүр монгол хэлээр ном уншдаг. Ээж минь өглөө бүр сүүтэй цай чанадаг."""

    # Held out: none of these sentences are in mongolian_train.
    mongolian_test = [
        "Сайн байна уу? Миний нэр Тэмүүжин.",
        "Өнөөдөр цаг агаар сайхан байна.",
        "Бид маргааш Хархорин руу явна.",
        "Энэ ном маш сонирхолтой.",
        "Хүүхдүүд сургууль руу явж байна.",
        "Баярлалаа, баяртай!",
    ]
setup:
  - say: Two tokenizers made with 1.5's code. One is trained on an English story, the other on a few Mongolian paragraphs, with 40 merges each.
    type: "en_merges, en_vocab = train(english, 296)\nmn_merges, mn_vocab = train(mongolian_train, 296)\n\n"
    pause: 2.8
  - say: "Fertility is how many tokens a tokenizer spends per word. Here's the stub. Your turn: count tokens and words over the whole test set."
    type_file: fertility/stub.py
    pause: 1.2
tests: fertility/tests.py
solution: fertility/solution.py
alternatives:
  - fertility/alternatives/loop.py
  - fertility/alternatives/joined.py
hints:
  logic: |
    for each sentence in sentences:
      encode it with the merges given → count the ids   (tokens)
      split it at spaces → count the words
    add the counts up over ALL the sentences
      → return total tokens ÷ total words
  tools: |
    tokens in one sentence
      encode(s, m)     the ids for s with merges m (defined above)
      len(x)           how many items x has
    words in one sentence
      s.split()        cuts s into words at spaces
    add up over all sentences
      sum(items)       adds up the items
      plain arithmetic: /
  nudge: Two totals over all the sentences. How many tokens does `encode` produce, and how many words are there? And which tokenizer should `encode` use?
  concept: "`encode(s, merges)` returns one sentence's token ids, so its length is the token count. `s.split()` cuts a sentence into words at spaces. Sum both over all the sentences, then divide, so long sentences weigh more."
  example: |
    Use the merges `AB = {(208, 176): 256, (208, 177): 257, (256, 257): 258}`, which turn `аб` into one token, on `["аб аб", "аб"]`:

    - `"аб аб"`: 9 bytes (each `а` and `б` is 2 bytes, plus the space). With `AB`, each `аб` becomes one token, so the ids are `[258, 32, 258]`: 3 tokens, 2 words.
    - `"аб"`: 1 token, 1 word.
    - Totals: `tokens = 4`, `words = 3`, so fertility is `4 / 3 ≈ 1.333`.
    - Averaging the two sentences' ratios instead gives `(1.5 + 1) / 2 = 1.25`. That's the wrong answer.

    With no merges, the same `"аб аб"` stays at 9 tokens for 2 words, which is `4.5`.
  skeleton: |
    tokens = sum(len(______(s, ______)) for s in sentences)
    words = sum(len(s._____()) for s in sentences)
  solution: |
    tokens = sum(len(encode(s, merges)) for s in sentences)
    words = sum(len(s.split()) for s in sentences)
on_solve: "11.5 tokens per word with no Mongolian merges: one per byte, two per letter. The English-trained tokenizer is no better. 40 merges from a few paragraphs nearly halve it. Step 2: where those extra tokens come from."
mistakes:
  - { file: fertility/mistakes/placeholders.py, fails: tokens_per_word, says: "`tokens` is still `...`" }
  - { file: fertility/mistakes/characters.py, fails: tokens_per_word, says: "which is characters per word" }
  - { file: fertility/mistakes/bytes.py, fails: tokens_per_word, says: "which is bytes per word" }
  - { file: fertility/mistakes/per_sentence.py, fails: tokens_per_word, says: "That's tokens per *sentence*" }
  - { file: fertility/mistakes/global_merges.py, fails: tokens_per_word, says: "not the merges passed in" }
  - { file: fertility/mistakes/first_only.py, fails: all_sentences, says: "Only one of the two sentences got counted" }
```

### Count the byte pieces

```challenge
id: partial
file: audit.py
minutes: 20
start: { from: fertility }
setup:
  - say: With no merge for a letter, a byte-level tokenizer falls back to the letter's raw bytes, two tokens that mean nothing alone.
    type_file: partial/stub.py
    pause: 2.6
  - say: "Your turn: spot the pieces that aren't whole characters, then measure their share."
    pause: 1.0
tests: partial/tests.py
solution: partial/solution.py
alternatives:
  - partial/alternatives/else_branch.py
  - partial/alternatives/bare_except.py
  - partial/alternatives/counter.py
hints:
  logic: |
    is_partial(piece):
      try to decode piece as UTF-8   (no replacing, let it fail)
      decoding fails?
        yes → True    (bytes stop in the middle of a character)
        no  → False
    partial_share:
      for each piece in pieces:
        is_partial?
          yes → count it
      → return partial count ÷ number of pieces
  tools: |
    can these bytes be read as text?
      b.decode(enc)       bytes to text; raises on bad bytes
      try: … except E:    runs the code, catches error E
    count the partial pieces
      sum(items)          adds up items; True counts as 1
    out of all pieces
      len(x)              how many items x has
      plain arithmetic: /
  nudge: A piece is partial when it can't be read as text by itself. What does Python do when you decode bytes that stop in the middle of a character?
  concept: "`piece.decode(\"utf-8\")` raises UnicodeDecodeError on bytes that aren't complete characters. Catch it and return True; if decoding works, return False. The share is the number of partial pieces divided by the number of pieces."
  example: |
    Trace it on `"а б"`, where the only merge turns `а` (`b"\xd0\xb0"`) into one token:

    - Encoded, the pieces are `b"\xd0\xb0"`, `b" "`, `b"\xd0"`, `b"\xb1"`. The `б` never got a merge, so it's two single-byte tokens.
    - `is_partial` on each: `b"\xd0\xb0"` decodes to `"а"`, so `False`. `b" "` decodes, `False`. `b"\xd0"` raises, `True`. `b"\xb1"` raises, `True`.
    - Partial pieces: 2. Total pieces: 4. `partial_share` is `2 / 4 = 0.5`.

    Check: the print line multiplies by 100 itself, so you should return `0.5`, not `50`.
  skeleton: |
    try:
        piece.______("utf-8")
    except __________________:
        return ____
    return _____
    return sum(__________(p) for p in pieces) / len(______)
  solution: |
    try:
        piece.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False
    return sum(is_partial(p) for p in pieces) / len(pieces)
on_solve: "Look at Өнөөдөр, 'today': the capital Ө never got a merge, so it comes out as two '�' pieces. The model has to learn to put those bytes back together from context, extra work that a merged letter doesn't need. Step 3: how much of a vocabulary is Cyrillic at all."
mistakes:
  - { file: partial/mistakes/placeholders.py, fails: is_partial, says: "`is_partial` returns None" }
  - { file: partial/mistakes/by_length.py, fails: is_partial, says: "Length alone doesn't tell you" }
  - { file: partial/mistakes/high_byte.py, fails: is_partial, says: "A letter can take several bytes" }
  - { file: partial/mistakes/replace_never_fails.py, fails: is_partial, says: "decoding never fails" }
  - { file: partial/mistakes/backwards.py, fails: is_partial, says: "Every answer is backwards" }
  - { file: partial/mistakes/count_not_share.py, fails: share, says: "the number of partial tokens" }
  - { file: partial/mistakes/percent.py, fails: share, says: "a percentage" }
```

### Scan a vocabulary for Cyrillic

```challenge
id: cyrillic
file: audit.py
minutes: 20
start: { from: partial }
setup:
  - say: The last measure is how much of a vocabulary is spent on Cyrillic at all. Llama 3.1 has 127,744 merged tokens. How many contain a Cyrillic letter?
    type_file: cyrillic/stub.py
    pause: 2.8
  - say: "Your turn: recognise Cyrillic, Mongolian letters included, and compute the share of merged tokens that contain it."
    pause: 1.0
tests: cyrillic/tests.py
solution: cyrillic/solution.py
alternatives:
  - cyrillic/alternatives/char_range.py
  - cyrillic/alternatives/loop.py
  - cyrillic/alternatives/range_ids.py
hints:
  logic: |
    has_cyrillic(text):
      for each character in text:
        its code point is in U+0400…U+04FF?
          yes → return True
      no character matched → return False
    cyrillic_share(vocab):
      keep only the merged tokens   (id 256 or more)
      turn each one's bytes into text   (half a letter → �)
      count those that pass has_cyrillic
        → return that count ÷ number of merged tokens
  tools: |
    is any character in the block?
      ord(ch)             the code point (a number) of ch
      lo <= x <= hi       True if x is between lo and hi
      any(items)          True if at least one item is True
    the merged tokens, as text
      [e for k in d if c] a list of e for the keys k that pass c
      b.decode(enc, errors=mode)
                          bytes to text; mode chooses what
                          happens to bytes that aren't valid
    share of them with Cyrillic
      sum(items)          adds up items; True counts as 1
      len(x)              how many items x has
  nudge: Unicode keeps Cyrillic in one block of code points. Does а…я cover Ө and Ү? For the share, which ids are merged tokens, and how do you decode half a letter without crashing?
  concept: "Cyrillic is the block U+0400 to U+04FF, so `0x0400 <= ord(ch) <= 0x04FF` checks one character and `any(...)` checks a whole token. Merged tokens have ids 256 and up. `decode(\"utf-8\", errors=\"replace\")` turns half a letter into '�' instead of raising."
  example: |
    `has_cyrillic`:

    - `"ж"` is U+0436, inside `0x0400`–`0x04FF`: `True`.
    - `"ө"` is U+04E9. It's past `я` (U+044F) but still inside the block: `True`. A check for `а`…`я` alone would miss it.
    - `"hello"`, `"αβγ"`, `"中文"` and `""` have no code point in the block: `False`.

    `cyrillic_share`, with a vocab where 256 is `b"\xd0\xb6"`, 257 is `b"th"`, 258 is `b"\xd0"` and 259 is `b"\xd0\xb0\xd1\x80"`:

    - Ids below 256 are skipped. The merged tokens are 256 to 259, four of them.
    - Decoded: `"ж"`, `"th"`, `"�"` (half a letter, replaced instead of raising), `"ар"`.
    - Contain Cyrillic: `"ж"` and `"ар"`. That's 2 of 4, so the share is `0.5`.
    - If you counted all 260 ids, you'd get `2 / 260`. The single bytes only dilute the share.
  skeleton: |
    return ___(0x0400 <= ___(ch) <= 0x04FF for ch in text)
    merged = [vocab[i].______("utf-8", errors="_______") for i in vocab if i >= ___]
  solution: |
    return any(0x0400 <= ord(ch) <= 0x04FF for ch in text)
    merged = [vocab[i].decode("utf-8", errors="replace") for i in vocab if i >= 256]
on_solve: "0% for the English tokenizer, 95% for the Mongolian one. Point the same function at Llama 3.1's 127,744 merges and you have this lesson's systems metric."
mistakes:
  - { file: cyrillic/mistakes/placeholders.py, fails: has_cyrillic, says: "`has_cyrillic` returns None" }
  - { file: cyrillic/mistakes/russian_only.py, fails: has_cyrillic, says: "sit outside а…я" }
  - { file: cyrillic/mistakes/lowercase_only.py, fails: has_cyrillic, says: "sit outside а…я" }
  - { file: cyrillic/mistakes/non_ascii.py, fails: has_cyrillic, says: "Non-ASCII covers every script" }
  - { file: cyrillic/mistakes/all_chars.py, fails: has_cyrillic, says: "One Cyrillic letter anywhere is enough" }
  - { file: cyrillic/mistakes/strict_decode.py, fails: share, says: "Decoding failed on a merged token" }
  - { file: cyrillic/mistakes/raw_bytes.py, fails: share, says: "`merged` holds bytes" }
  - { file: cyrillic/mistakes/whole_vocab.py, fails: share, says: "the 256 single bytes too" }
```

### Audit Llama 3.1

```replay
id: llama31_audit
intro: "Now the real audit on your machine: load Llama 3.1's tokenizer with transformers and run the same three measures on a larger Mongolian sample and its English translation. It runs in under a minute on a CPU. Here's the shape of the output."
log: replays/llama31-audit.jsonl
speed: 40
recorded:
  hardware: Laptop CPU
  illustrative: true
  note: Sample output written for the course to show what the audit prints. These are not measured Llama 3.1 numbers, and the splits are examples. Run it to get yours.
chart:
  title: Tokens per word
  flag_above: 2.0
  x_labels: [English, Mongolian]
  caption: The same sentences in two languages. Mongolian costs about 2.6× as many tokens per word, in context, compute and price.
summary:
  - { label: Mongolian fertility, value: 3.38 tok/word }
  - { label: vs English, value: 2.58× }
  - { label: Cyrillic merges, value: 4.6% of the vocab }
run_yourself: |
  # on your machine (CPU is fine), from the course repo
  # needs a Hugging Face account with access to meta-llama/Llama-3.1-8B
  pip install transformers
  python audit_tokenizer.py --tokenizer meta-llama/Llama-3.1-8B \
    --sample data/mn_sample.txt --compare data/en_sample.txt
```

### Record your metrics

```metrics
id: record
intro: Llama 3.1's fertility on Mongolian and the share of its merges that contain Cyrillic, from your own audit run.
```

## Hints

- Compare against the *same* sentences in English. Then the fertility ratio measures the tokenizer, not the content.
- `tokenizer.convert_ids_to_tokens` shows byte-level tokens through GPT-2's byte-to-character mapping, so Cyrillic looks like `Ð°`. Decode each id with `tokenizer.decode([i])` to see real text.
- Count words with `str.split()`. Punctuation stays attached to words, which is fine as long as both languages are counted the same way.
- Save ten common Mongolian words and how Llama 3.1 splits them. In 4B.1 you train a Mongolian tokenizer and compare against this list.

## Done when

- [ ] fertility works for any tokenizer you pass it {#fertility auto=challenge:fertility}
- [ ] is_partial and partial_share find the byte pieces {#partial auto=challenge:partial}
- [ ] cyrillic_share counts Mongolian letters as Cyrillic {#cyrillic auto=challenge:cyrillic}
- [ ] Llama 3.1's fertility and Cyrillic coverage recorded {#metrics_recorded auto=metrics}
