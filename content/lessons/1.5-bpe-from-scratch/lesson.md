---
id: "1.5"
title: BPE from scratch
summary: Write byte-pair encoding in plain Python, from counting pairs to a tokenizer that round-trips English, Mongolian and emoji.
phase: "1"
hours: 4
type: build
status: ready
protected: false
cut: null
prerequisites:
  - "1.3"
papers:
  - title: Neural Machine Translation of Rare Words with Subword Units
    authors: Sennrich et al.
    year: 2016
    url: https://arxiv.org/abs/1508.07909
quality_metric:
  id: roundtrip_ok
  label: Encode/decode round trip on the test set
  unit: "%"
  note: Eight lines in English, Mongolian Cyrillic, Chinese, emoji and code, none of which the story contains.
  inputs:
    - { id: roundtrip_ok, label: Round-trip lines OK (%), hint: "Printed by step 3: \"round trip\"" }
  compute: value
  reference: 100
  pass: { min: 100 }
systems_metric:
  id: encode_mb_s
  label: Encoding speed
  unit: MB/s
  note: Your pure-Python encoder in the browser. tiktoken, written in Rust, encodes at tens of MB/s.
  inputs:
    - { id: encode_mb_s, label: Encoding speed (MB/s), hint: "Printed by step 3: \"encoding speed\"" }
  compute: value
  reference: 0.09
  pass: { min: 0.01 }
quiz:
  - id: why_bytes
    q: Why does byte-level BPE start from the 256 byte values instead of from characters?
    answer: 2
    options:
      - text: Bytes compress better than characters
        why: The starting alphabet barely affects compression. The merges do the compressing.
      - text: Characters would make the vocabulary too small
        why: The opposite. Unicode has about 150,000 characters, so starting from characters would make the base vocabulary huge.
      - text: Any text in any script is some sequence of bytes, so nothing is ever unknown
        why: Mongolian, Chinese, emoji or a binary blob all encode, even with zero merges for them. The cost is length. Unmerged text takes one token per byte, which 1.6 measures for Mongolian.
      - text: GPUs can only process 8-bit numbers
        why: Token ids index an embedding table and can be any size. Llama 3's go up to 128,255.
  - id: merge_order
    q: When encoding new text, why apply merges in the order training learned them, not most frequent pair first?
    answer: 0
    options:
      - text: Later merges were learned on top of earlier ones, so a different order builds tokens training never saw
        why: Training learned ' th' from ' t' + 'h', after ' t' existed. Replaying merges in their original order reproduces exactly the tokens training made, and the same text always gets the same ids.
      - text: It's faster
        why: Both orders count pairs on every pass. The reason is correctness, not speed.
      - text: Most-frequent-first would lose bytes
        why: Any order of merges keeps every byte. decode would still round-trip; you'd just get different, inconsistent tokens.
      - text: Frequencies in a short text are too noisy
        why: Even in a long text, the counts there are a different question from what training learned. Order matters because merges build on each other.
---

## Goal

Write **byte-pair encoding** in plain Python. Count which neighbouring tokens appear together most, fuse that pair into a new token, and repeat until you have a vocabulary. Then **encode** new text by replaying the merges in order, and **decode** it back. Your tokenizer should round-trip English, Mongolian Cyrillic, Chinese and emoji, even though it was trained on one short English story.

## Learn

```card
type: concept
id: merges_compress
eyebrow: The idea
title: Every merge makes the text shorter.
visual:
  kind: bars
  max: 800
  bars:
    - { label: 256 (bytes), value: 792, text: "792" }
    - { label: "276", value: 548, text: "548" }
    - { label: "306", value: 414, text: "414" }
    - { label: "356", value: 304, text: "304" }
body: |
  Start with the text as raw **bytes**, ids 0–255. Find the pair of neighbours that appears most often, say `e` followed by a space, and give it a new id, 256. Replace every occurrence, then count again. Each **merge** shortens the text and adds one token to the vocabulary. The bars show this lesson's 792-byte story in tokens, at vocab sizes from 256 up to 356.
```

```card
type: predict
id: predict_merge
eyebrow: Predict the output
title: Merge "aa" into 256. What prints?
code: |
  ids = list("aaabdaaabac".encode("utf-8"))   # a=97 b=98 c=99 d=100
  print(merge(ids, (97, 97), 256))
options:
  - text: "[256, 97, 98, 100, 256, 97, 98, 97, 99]"
    correct: true
    why: Going left to right, the first two a's merge, then the third a has no partner left. The same happens in the second "aaab". Merges never overlap.
  - text: "[256, 256, 98, 100, 256, 256, 98, 97, 99]"
    why: That merges "aa" twice inside "aaa", reusing the middle a. Once an id is part of a merge, it's used up.
  - text: "[97, 256, 98, 100, 97, 256, 98, 97, 99]"
    why: That merges from the right. merge walks left to right, so the first two a's pair up.
  - text: "[256, 98, 100, 256, 98, 97, 99]"
    why: That treats "aaa" as one token. Each merge fuses exactly two ids into one.
hint: Walk left to right. When two a's merge, both are used up, and the next comparison starts after them.
```

```card
type: fill_blank
id: count_pairs
eyebrow: Fill the blanks
title: Count every pair of neighbours.
code: |
  counts = {}
  for pair in zip(ids, [[0]]):
      counts[pair] = counts.get(pair, [[1]]) + 1
tokens: ["ids[1:]", ids, "ids[:-1]", "0", "1"]
answer: ["ids[1:]", "0"]
tips:
  ids: zip(ids, ids) pairs each id with itself. You want each id with the *next* one.
  "ids[:-1]": That's the list without its last id, which lines each id up with itself again.
  "1": A pair seen for the first time would start at 2. `get` should fall back to 0, then the + 1 counts it.
why_ok: "`zip(ids, ids[1:])` walks the list beside itself shifted by one, so it yields every pair of neighbours, overlapping. `get(pair, 0) + 1` counts each one, new or not."
why_wrong: "Pair each id with the one after it, and start a new pair's count at zero."
hint: Shift the list by one position and walk it side by side with the original.
```

```card
type: calc
id: utf8_bytes
eyebrow: Do the math
title: How many bytes is "Сайн байна уу?" in UTF-8?
given:
  - { label: Text, value: "Сайн байна уу? (11 Cyrillic letters, 2 spaces, 1 ?)" }
  - { label: UTF-8, value: "ASCII 1 byte · Cyrillic 2 · Chinese 3 · emoji 4" }
unit: bytes
answer: { min: 25, max: 25 }
placeholder: "0"
why_ok: 11 × 2 + 3 × 1 = 25 bytes. A tokenizer with no Cyrillic merges, like yours after training on an English story, needs 25 tokens for these 14 characters.
why_wrong: Each Cyrillic letter is 2 bytes; spaces and the question mark are 1 each.
hint: Count the Cyrillic letters and the ASCII characters separately.
```

```card
type: reorder
id: one_round
eyebrow: Put it in order
title: Arrange one round of BPE training.
lines:
  - ids = list(text.encode("utf-8"))
  - stats = get_stats(ids)
  - pair = max(stats, key=stats.get)
  - ids = merge(ids, pair, 256)
shuffle: [2, 0, 3, 1]
why_ok: "Bytes first, then count the pairs, pick the most frequent, and fuse it everywhere. Training repeats the last three lines, with a new id each time."
why_wrong: "Follow the names: `stats` needs `ids`, `pair` needs `stats`, and `merge` needs `pair`."
hint: Each line can only use names an earlier line created.
```

```card
type: quiz
ref: why_bytes
```

```card
type: quiz
ref: merge_order
```

## Build steps

### Count pairs and merge them

```challenge
id: pairs
file: bpe.py
minutes: 25
start:
  code: |
    # A short story to train on. Every token you make comes from these bytes.
    text = """Once upon a time, there was a little bird named Tumen. Tumen lived in a tall tree by a quiet river. Every morning, Tumen sang a happy song for the other birds in the tree.
    One day, the wind was very strong. The wind blew the nest out of the tree and into the river. Tumen was sad. He did not know what to do.
    A kind old turtle saw the nest in the water. "Do not be sad, little bird," said the turtle. "I will help you." The turtle swam to the nest and pushed it to the shore.
    Tumen was so happy! He thanked the turtle and carried the nest back up the tree, one stick at a time. That night, Tumen sang his best song for the turtle.
    From that day on, the little bird and the old turtle were best friends. They played by the river every day, and when the wind was strong, they helped each other."""
setup:
  - say: BPE starts from raw bytes. Here's a short string as bytes. Every character is one number from 0 to 255.
    type: "\nids = list(\"aaabdaaabac\".encode(\"utf-8\"))\nprint(ids)\n\n"
    pause: 2.6
  - say: "The whole algorithm is two moves: count which neighbours appear together most, and fuse that pair into one new token. Here's the stub. Your turn: count the pairs, then write the merge."
    type_file: pairs/stub.py
    pause: 1.2
tests: pairs/tests.py
solution: pairs/solution.py
alternatives:
  - pairs/alternatives/range_loop.py
  - pairs/alternatives/if_in.py
  - pairs/alternatives/slice_compare.py
hints:
  nudge: Neighbours overlap. In [1, 2, 3], the 2 belongs to both (1, 2) and (2, 3). For merge, what has to be true at position i before you write idx?
  concept: "`zip(ids, ids[1:])` walks the list beside itself shifted by one, giving every pair of neighbours. `counts.get(pair, 0) + 1` adds one whether or not you've seen the pair before. In merge, position i starts the pair when ids[i] and ids[i + 1] match its two ids, and there has to *be* an i + 1."
  skeleton: |
    for pair in zip(___, ___[1:]):
        counts[pair] = counts.___(pair, _) + 1
    if i < len(ids) - _ and ids[i] == pair[_] and ids[i + 1] == pair[_]:
  solution: |
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
on_solve: "(97, 97), 'aa', four times. Notice that 'aaab' became [256, 97, 98]: merging goes left to right and never overlaps. Step 2 is unlocked: repeat count-and-merge to learn a vocabulary."
mistakes:
  - { file: pairs/mistakes/loop_placeholder.py, fails: counts_pairs, says: "still loops over `...`" }
  - { file: pairs/mistakes/count_placeholder.py, fails: counts_pairs, says: "The counts are still `...`" }
  - { file: pairs/mistakes/set_not_add.py, fails: counts_pairs, says: "You set the count instead of adding to it" }
  - { file: pairs/mistakes/self_pairs.py, fails: counts_pairs, says: "each id paired with itself" }
  - { file: pairs/mistakes/even_only.py, fails: counts_pairs, says: "Neighbours overlap" }
  - { file: pairs/mistakes/past_end.py, fails: counts_pairs, says: "reads past the end of `ids`" }
  - { file: pairs/mistakes/if_placeholder.py, fails: merges_pair, says: "Python treats as True" }
  - { file: pairs/mistakes/no_skip.py, fails: merges_pair, says: "skip past both ids" }
  - { file: pairs/mistakes/first_only.py, fails: merges_pair, says: "you merged (1, 3) too" }
  - { file: pairs/mistakes/no_bounds.py, fails: merges_pair, says: "there's no `ids[i + 1]`" }
```

### Train a vocabulary

```challenge
id: train
file: bpe.py
minutes: 25
start: { from: pairs }
setup:
  - say: Count, merge, repeat. Each round the most frequent pair becomes a new token, and the vocab remembers which bytes it stands for.
    type_file: train/stub.py
    pause: 2.6
  - say: "Your turn: pick the pair, give it the next id, and record its bytes. The script trains 20 merges on the story."
    pause: 1.2
tests: train/tests.py
solution: train/solution.py
alternatives:
  - train/alternatives/items_max.py
  - train/alternatives/len_vocab.py
  - train/alternatives/unpacked.py
hints:
  nudge: Three blanks. Which pair wins, which id comes after the last one used, and which bytes does the new token stand for?
  concept: "`max(stats, key=stats.get)` returns the pair whose *count* is largest. Ids 0–255 are the bytes, so the i-th merge gets 256 + i. A merged token stands for its first part's bytes followed by its second part's, `vocab[a] + vocab[b]`, which works even when a part is itself a merged token."
  skeleton: |
    pair = max(stats, key=_____.___)
    idx = ___ + i
    vocab[idx] = vocab[pair[_]] + vocab[pair[_]]
  solution: |
    pair = max(stats, key=stats.get)
    idx = 256 + i
    vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
on_solve: "Read the first merges: ' t', 'e ', ' th', and by the sixth, ' the '. BPE found English's most common word by itself, starting from bytes. Step 3 is unlocked: use the merges on new text."
mistakes:
  - { file: train/mistakes/placeholders.py, fails: first_merge, says: "`pair` is still `...`" }
  - { file: train/mistakes/max_pairs.py, fails: first_merge, says: "compares the pairs themselves, not their counts" }
  - { file: train/mistakes/least_frequent.py, fails: first_merge, says: "Which pair should win?" }
  - { file: train/mistakes/idx_from_zero.py, fails: new_ids, says: "ids 0–255 are already the 256 bytes" }
  - { file: train/mistakes/idx_constant.py, fails: new_ids, says: "Every merge got id 256" }
  - { file: train/mistakes/bytes_of_pair.py, fails: new_ids, says: "only takes numbers up to 255" }
  - { file: train/mistakes/vocab_placeholder.py, fails: vocab_bytes, says: "`vocab[idx]` is still `...`" }
  - { file: train/mistakes/first_part_only.py, fails: vocab_bytes, says: "only the first part" }
```

### Encode and decode

```challenge
id: encode
file: bpe.py
minutes: 30
start: { from: train }
setup:
  - say: Training is done. Encoding new text replays the merges, and decoding looks each token's bytes back up.
    type_file: encode/stub.py
    pause: 2.6
  - say: "The test set mixes English, Mongolian, Chinese, emoji and code, none of it in the story. Your turn: write decode, then pick the right pair in encode."
    pause: 1.2
tests: encode/tests.py
solution: encode/solution.py
alternatives:
  - encode/alternatives/bytes_concat.py
  - encode/alternatives/rank_function.py
  - encode/alternatives/sorted_first.py
hints:
  nudge: "decode: each id stands for some bytes, so how do you glue bytes together? encode: of the pairs in this text, which one would training have merged first?"
  concept: "Join every token's bytes into one bytes object, then decode it once, so a character split across two tokens comes back whole. In encode, rank each pair by its id in `merges` (a smaller id was learned earlier) and give pairs that were never merged infinity, so they never win."
  skeleton: |
    data = _"".join(_____[idx] for idx in ids)
    pair = min(stats, key=lambda p: merges.___(p, float("___")))
  solution: |
    data = b"".join(vocab[idx] for idx in ids)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
on_solve: "8 of 8 lines round-trip, in scripts the story never contained. Bytes make that free: every text is bytes, so nothing is unknown. Now look at the Mongolian line: 14 characters, 25 tokens. With no Cyrillic merges, every letter costs 2. Lesson 1.6 measures that for Llama 3.1's real tokenizer."
mistakes:
  - { file: encode/mistakes/placeholders.py, fails: decode_bytes, says: "`data` is still `...`" }
  - { file: encode/mistakes/list_of_bytes.py, fails: decode_bytes, says: "a list of byte strings" }
  - { file: encode/mistakes/str_join.py, fails: decode_bytes, says: "joins strings, but each vocab entry is bytes" }
  - { file: encode/mistakes/raw_bytes.py, fails: decode_bytes, says: "only works for raw bytes" }
  - { file: encode/mistakes/per_token.py, fails: decode_bytes, says: "You decoded each token on its own" }
  - { file: encode/mistakes/pair_placeholder.py, fails: encode_earliest, says: "nothing got merged" }
  - { file: encode/mistakes/most_frequent.py, fails: encode_earliest, says: "you merged the most frequent pair" }
  - { file: encode/mistakes/key_error.py, fails: encode_earliest, says: "raises a KeyError" }
```

### Record your metrics

```metrics
id: record
intro: The round-trip score and the encoding speed your tokenizer printed in step 3.
```

## Hints

- Real tokenizers split text into words first (GPT-2 and Llama 3 use a regex), so merges never cross from one word into the next, and "dog." and "dog!" share the token "dog".
- Encoding recounts every pair on every pass, which is why yours manages about 0.1 MB/s. Production tokenizers keep a priority queue of pairs and run in Rust.
- `errors="replace"` in decode matters for *generation*. A model can emit half of a multi-byte character, and without it decoding that stream would raise.
- Save `merges` in order. The order is the tokenizer: the same merges in a different order encode text differently.

## Done when

- [ ] get_stats and merge handle overlaps and the end of the list {#pairs auto=challenge:pairs}
- [ ] train learns ' the ' from bytes {#train auto=challenge:train}
- [ ] Every line of the test set round-trips {#encode auto=challenge:encode}
- [ ] Round trip and encoding speed recorded {#metrics_recorded auto=metrics}
