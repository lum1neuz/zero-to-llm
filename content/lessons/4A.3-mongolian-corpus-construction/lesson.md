---
id: "4A.3"
title: Mongolian corpus construction
summary: Turn scraped Mongolian web text into a clean training corpus. Fix invisible Unicode damage, filter out Russian and Kazakh, and remove near-duplicates with MinHash.
phase: "4A"
hours: 10
type: research
status: ready
protected: true
cut: null
prerequisites:
  - 4A.2
  - "1.6"
fallbacks:
  "1.6": Skip the comparison with the audit. Use the tokens-per-word figure this lesson's pipeline prints for Llama 3.1's tokenizer.
papers:
  - title: Deduplicating Training Data Makes Language Models Better
    authors: Lee et al.
    year: 2022
    url: https://arxiv.org/abs/2107.06499
  - title: "On the resemblance and containment of documents"
    authors: Broder
    year: 1997
quality_metric:
  id: dedup_pct
  label: Near-duplicates removed
  unit: "%"
  note: The share of documents MinHash removes, out of those that reach the dedup stage. Under 5% usually means dedup isn't catching mirrors. Over 40% usually means the threshold is too low.
  inputs:
    - { id: dedup_pct, label: Near-duplicates removed (%), hint: The MinHash line of your pipeline's report }
  compute: value
  reference: 21.0
  pass: { min: 5.0, max: 40.0 }
systems_metric:
  id: corpus_tokens_m
  label: Clean corpus size
  unit: M tokens
  note: Counted with Llama 3.1's tokenizer, untouched. 4A.4 trains on a 20M-token sample of it, so anything above 20M is enough.
  inputs:
    - { id: corpus_tokens_m, label: Clean corpus (M tokens), hint: The last line of your pipeline's report }
  compute: value
  reference: 191
  pass: { min: 20 }
quiz:
  - id: script_not_language
    q: A filter keeps every document whose letters are at least 80% Cyrillic. What gets through?
    answer: 0
    options:
      - text: Mongolian, and also Russian and Kazakh
        why: All three are written in Cyrillic. A script test can't tell them apart. Mongolian's ө and ү rule out Russian, and Kazakh's extra letters (ә, қ, ң, ұ, і…) rule out Kazakh.
      - text: Only Mongolian, because 80% is strict
        why: The threshold measures the script, not the language. Russian news is 100% Cyrillic and sails through.
      - text: Mongolian and Russian, but not Kazakh
        why: Kazakh is Cyrillic too. It even shares ө and ү with Mongolian, which is why it needs its own check.
  - id: why_near_dups
    q: Why remove near-duplicates, and not only exact copies, before training?
    answer: 1
    options:
      - text: To save disk space
        why: That's a side effect. The real costs of duplicates land on the model and on your eval.
      - text: Mirrored articles get memorised and over-weighted, and copies in the held-out set flatter the eval
        why: News is mirrored across dozens of sites with a different footer each time, so exact matching misses it. Lee et al. found that near-duplicate removal cut memorised output and gave more honest held-out numbers.
      - text: Near-duplicates confuse the tokenizer
        why: The tokenizer here is fixed, Llama 3.1's, untouched. Duplicates don't change it. They change what the model sees most often.
  - id: split_after_dedup
    q: When should you split off the held-out set that 4A.4 measures perplexity on?
    answer: 2
    options:
      - text: Before any cleaning, so it reflects the raw web
        why: Then a mirror of a held-out article can sit in the training set, and the model is graded on text it has effectively seen.
      - text: It doesn't matter, as long as the split is random
        why: A random split of a corpus full of near-copies puts copies on both sides. Held-out perplexity then measures memorisation.
      - text: After dedup, from the clean corpus
        why: Once near-duplicates are gone, no held-out document has a twin in training, so held-out perplexity measures how well the model knows Mongolian.
---

## Goal

Build the Mongolian corpus that every run in 4A and 4B trains on. Web text arrives with invisible damage (letters typed in pieces, Latin letters posing as Cyrillic), mixed with Russian, Kazakh, English, menus and page counters, and mirrored across sites with a new footer each time. Write the four stages yourself on twelve real-looking documents: **normalize**, **filter** by language and quality, compare documents with **word shingles and Jaccard**, and find near-duplicates with **MinHash**. Then run the same pipeline over a full crawl.

## Learn

```card
type: concept
id: why_clean
eyebrow: Why bother
title: Of 1.2M scraped Mongolian documents, 700k are worth training on.
visual:
  kind: bars
  max: 1250
  bars:
    - { label: scraped, value: 1204, text: "1,204k" }
    - { label: filtered, value: 998, text: 998k }
    - { label: exact dedup, value: 886, text: 886k }
    - { label: near dedup, value: 700, text: 700k }
body: |
  Mongolian web text is scarce, so it's tempting to keep everything. But a crawl labelled "Mongolian" is full of Russian and Kazakh (same alphabet), cookie banners and page counters, and the same news story mirrored across twenty sites. Each stage removes something that would teach the model the wrong thing, or make your eval lie.
```

```card
type: predict
id: composed
eyebrow: Predict the output
title: Two ways to write й. What prints?
code: |
  a = "й"                          # typed on a Mongolian keyboard
  b = "и" + "\N{COMBINING BREVE}"  # pasted from a PDF
  print(a == b, len(a), len(b))
options:
  - text: "True 1 1"
    why: They look identical on screen, but Python compares code points, and b has two.
  - text: "False 1 2"
    correct: true
    why: a is one code point, U+0439. b is и (U+0438) followed by a combining breve (U+0306). Same letter on screen, different strings, so a tokenizer and a dedup hash see different words. NFC normalization composes b into a.
  - text: "True 1 2"
    why: If they had different lengths they couldn't be equal. Python's `==` compares code points one by one, with no Unicode normalization.
  - text: "False 2 2"
    why: a is a single precomposed character, U+0439, so its length is 1.
hint: How many code points does each string hold?
```

```card
type: calc
id: jaccard_calc
eyebrow: Do the math
title: Two news pages, cut into 3-word shingles. What's their Jaccard similarity?
given:
  - { label: Shingles in page A, value: "14" }
  - { label: Shingles in page B, value: "16" }
  - { label: Shingles in both, value: "12" }
unit: ""
answer: { min: 0.66, max: 0.67 }
placeholder: "0.00"
why_ok: "The union has 14 + 16 − 12 = 18 distinct shingles, so Jaccard is 12 / 18 = 0.67. Just under a 0.7 threshold. Pages this close are usually the same story with a different intro or footer."
why_wrong: Jaccard is shared over all the distinct shingles of both. Don't count the shared ones twice in the union.
hint: The union is A + B minus what they share.
```

```card
type: concept
id: minhash_trick
eyebrow: The trick
title: Two documents share a first shingle exactly as often as their Jaccard.
body: |
  Put every possible shingle in a random order and ask which of a document's shingles comes first. For two documents, the chance they pick the same one is exactly their Jaccard similarity. A seeded hash gives one random order per seed, and its smallest value over a document's shingles marks the first one. 64 seeds give a **signature** of 64 numbers. The share of positions where two signatures agree estimates Jaccard, without keeping the shingle sets around. Locality-sensitive hashing then buckets signatures, so you only ever compare documents that already agree on a whole band.
```

```card
type: reorder
id: one_doc
eyebrow: Put it in order
title: Arrange one document's trip from raw text to a dedup decision.
lines:
  - text = normalize(raw)
  - words = text.split()
  - sh = {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}
  - sig = [min(h(seed, s) for s in sh) for seed in range(64)]
  - dup = any(estimate(sig, s) >= 0.7 for s in sigs)
shuffle: [3, 0, 4, 2, 1]
why_ok: "Normalize first, so a Latin a can't make two copies look different. Then words, shingles, the signature, and the comparison with every signature kept so far."
why_wrong: "Follow the names. Each line uses what the line before it made."
hint: The first line takes the raw text. The last one needs a signature.
```

```card
type: quiz
ref: script_not_language
```

```card
type: quiz
ref: why_near_dups
```

```card
type: quiz
ref: split_after_dedup
```

## Build steps

### Normalize the text

```challenge
id: normalize
file: corpus.py
minutes: 30
start: { code: "import unicodedata\n\n" }
setup:
  - say: "Here are twelve documents as they came off the web: news, a mirror of the same news, Russian, Kazakh, English, a footer and a page counter. Some of the damage is invisible."
    type_file: normalize/raw_docs.py
    pause: 3.0
  - say: "Two kinds of invisible damage. Letters that arrived in pieces, like и plus a combining breve for й. And Latin letters that look exactly like Cyrillic ones, an a or an o typed on the wrong keyboard. Both have to be fixed before any filter or hash looks at the text."
    pause: 5.0
  - say: "Here's the stub, with a table of look-alikes. Your turn: write normalize."
    type_file: normalize/stub.py
    pause: 1.2
tests: normalize/tests.py
solution: normalize/solution.py
alternatives:
  - normalize/alternatives/nfkc.py
  - normalize/alternatives/loop.py
  - normalize/alternatives/regex.py
hints:
  logic: |
    put split-up letters back together (й from и + breve)
      → split the text into words on any whitespace
    for each word:
      has Cyrillic?
        yes → swap look-alike Latin letters for Cyrillic ones
        no  → leave it alone
    join the words with single spaces
      → return the text
  tools: |
    put split-up letters back together
      unicodedata.normalize(form, s)   recomposes letters in s
    split into words on any whitespace
      s.split()                        words, split on any whitespace
    fix look-alikes, only in Cyrillic words
      w.translate(table)               swaps characters using a table
      a if cond else b                 picks one of two values
      [expr for w in seq]              builds a list, one item per word
    join with single spaces
      sep.join(items)                  glues strings with sep between
  nudge: Two moves. Put the pieces of each letter back together, then go word by word and repair look-alikes only where the word is already Cyrillic.
  concept: "`unicodedata.normalize(\"NFC\", text)` composes letters (и + breve → й, o + diaeresis → ö). `text.split()` with no argument splits at any run of whitespace, tabs and non-breaking spaces included. `word.translate(LOOKALIKES)` swaps every look-alike in a word. `has_cyrillic(word)` tells you whether to."
  example: |
    Three small cases:

    - `"Сургуулийн"` is 11 characters: и followed by a combining breve. After recomposing it is 10 characters, the breve folded into a single й.
    - `"Малчид  өвөл бэл\tбаг".split()` gives `['Малчид', 'өвөл', 'бэл', 'баг']`. The double space, the non-breaking space and the tab are all gone.
    - The word `"хотoд"` has a Latin `o` (code point `0x6f`) in the middle. It contains Cyrillic, so its look-alike is swapped and it becomes `"хотод"`. The English word `"news"` has no Cyrillic, so it stays `"news"`. That is why you only repair words that are already Cyrillic.
  skeleton: |
    text = unicodedata.normalize("___", text)
    words = [w.translate(__________) if has_cyrillic(_) else w for w in text._____()]
  solution: |
    text = unicodedata.normalize("NFC", text)
    words = [w.translate(LOOKALIKES) if has_cyrillic(w) else w for w in text.split()]
on_solve: "Four documents changed, and none of them look any different. Doc 0's Latin a and o are now Cyrillic, doc 1 lost a double space, doc 4 a tab and a non-breaking space, and doc 5's two й are single characters again. Step 2 is unlocked. Decide what to keep."
mistakes:
  - { file: normalize/mistakes/placeholders.py, fails: composes, says: "`words` is still `...`" }
  - { file: normalize/mistakes/text_placeholder.py, fails: composes, says: "`text` is still `...`" }
  - { file: normalize/mistakes/no_nfc.py, fails: composes, says: "Which Unicode normal form composes them?" }
  - { file: normalize/mistakes/nfd.py, fails: composes, says: "That's NFD" }
  - { file: normalize/mistakes/split_space.py, fails: whitespace, says: "Two spaces in a row survive" }
  - { file: normalize/mistakes/no_translate.py, fails: lookalikes, says: "are Latin (U+0061, U+006F)" }
  - { file: normalize/mistakes/translate_all.py, fails: latin_words_kept, says: "should stay Latin" }
  - { file: normalize/mistakes/wrong_order.py, fails: order, says: "Compose first" }
```

### Filter by language and quality

```challenge
id: keep_filter
file: corpus.py
minutes: 40
start: { from: normalize }
setup:
  - say: "Normalized. Now decide what's Mongolian, and what's worth training on. Cyrillic alone won't do it: Russian and Kazakh are Cyrillic too."
    pause: 3.6
  - say: "Here's the stub: four rules, cheapest first. Each one returns False as soon as a document fails it."
    type_file: keep_filter/stub.py
    pause: 1.2
tests: keep_filter/tests.py
solution: keep_filter/solution.py
alternatives:
  - keep_filter/alternatives/ratios.py
  - keep_filter/alternatives/sets.py
  - keep_filter/alternatives/one_return.py
hints:
  logic: |
    fewer than 6 words?
      yes → drop it
    letters under half of the non-space characters?
      yes → drop it (junk)
    Cyrillic letters under 80% of the letters?
      yes → drop it
    no Mongolian-only letter (ө or ү) in it?
      yes → drop it (Russian)
    any Kazakh-only letter in it?
      yes → drop it
    no check tripped → keep it
  tools: |
    count words and characters
      len(x)                        how many items are in x
      s.replace(a, b)               copy of s with a swapped for b
    share of letters that pass a test
      sum(cond for c in seq)        counts the items that pass
      plain arithmetic: * <         share compared with a threshold
    does a letter from a set appear?
      any(cond for c in seq)        True if at least one passes
      x in collection               membership test
      not cond                      flips True and False
  nudge: "Four checks, each a single `if … : return False`: too short, mostly not letters, mostly not Cyrillic, not Mongolian. What's the denominator of each share?"
  concept: "Count words with `len(words)`. Junk has few letters among its non-space characters, `len(text.replace(\" \", \"\"))`. The Cyrillic share is out of the letters, not every character, so digits and spaces don't count against a sentence. Russian never uses ө or ү, so a Mongolian document has at least one of them. Kazakh shares ө and ү, so also drop anything with a letter from `KAZAKH_ONLY`."
  example: |
    Words, letters, non-space characters, and Cyrillic share of the letters for some of the twelve documents:

    - The footer `Бүх эрх хуулиар хамгаалагдсан.`: 4 words. Rule 1 drops it.
    - The page counter `1. 2. 3. … Үзсэн: 1520 | Сэтгэгдэл: 0`: 13 words, 14 letters out of 38 non-space characters, 0.37. Rule 2 drops it.
    - The English line: 12 words, 54 letters out of 55 (0.98), but 0 of them Cyrillic. Rule 3 drops it.
    - The Russian line: 11 words, 69 of 72 (0.96), 100% Cyrillic, but no ө or ү. Rule 4 drops it.
    - The Kazakh line: passes rules 1 to 3 and has a `ү`, but it also has `қ` and `і`, so the Kazakh-letter check drops it.
    - A Mongolian line like `Монгол Улсын Их Хурлын намрын ээлжит чуулган өнөөдөр нээлтээ хийлээ.`: 10 words, 58 of 59 (0.98), 100% Cyrillic, has ө, no Kazakh letters. Kept.
  skeleton: |
    if len(words) < ___:
        return False
    if len(letters) < ___ * len(text.replace(" ", "")):
        return False
    if sum(is_cyrillic(c) for c in letters) < ___ * len(_______):
        return False
    if not any(c in ______________ for c in letters) or any(c in ___________ for c in letters):
        return False
  solution: |
    if len(words) < 6:
        return False
    if len(letters) < 0.5 * len(text.replace(" ", "")):
        return False
    if sum(is_cyrillic(c) for c in letters) < 0.8 * len(letters):
        return False
    if not any(c in MONGOLIAN_ONLY for c in letters) or any(c in KAZAKH_ONLY for c in letters):
        return False
on_solve: "Seven of twelve survive. Each rule caught exactly what it was for: the footer was too short, the page counter was mostly digits, the English wasn't Cyrillic, the Russian had no ө or ү, and the Kazakh had і and қ. Step 3 is unlocked. Two of the seven survivors are the same story."
mistakes:
  - { file: keep_filter/mistakes/placeholders.py, fails: length, says: "and it's only 4 words" }
  - { file: keep_filter/mistakes/inverted.py, fails: length, says: "and it's only 4 words" }
  - { file: keep_filter/mistakes/no_junk.py, fails: junk, says: "it's a page counter, not text" }
  - { file: keep_filter/mistakes/no_cyrillic.py, fails: cyrillic, says: "What share of the letters should be Cyrillic" }
  - { file: keep_filter/mistakes/no_language.py, fails: language, says: "The Russian line is kept" }
  - { file: keep_filter/mistakes/no_kazakh.py, fails: language, says: "The Kazakh line is kept" }
  - { file: keep_filter/mistakes/le6.py, fails: keeps_mongolian, says: "it has exactly 6 words" }
  - { file: keep_filter/mistakes/cyr_all_chars.py, fails: keeps_mongolian, says: "or by every character?" }
  - { file: keep_filter/mistakes/both_letters.py, fails: keeps_mongolian, says: "Require ө *or* ү" }
  - { file: keep_filter/mistakes/all_cyrillic.py, fails: keeps_mongolian, says: "One brand name shouldn't sink a Mongolian sentence" }
```

### Compare documents with shingles

```challenge
id: shingles
file: corpus.py
minutes: 25
start: { from: keep_filter }
setup:
  - say: "Docs 0 and 1 are the same weather report, one with a source line on the end. Exact matching won't catch that. Cut each document into overlapping runs of 3 words, its shingles, and compare the sets."
    pause: 4.2
  - say: "Here's the stub. Write shingles and jaccard. The loop prints every pair above 0.2."
    type_file: shingles/stub.py
    pause: 1.2
tests: shingles/tests.py
solution: shingles/solution.py
alternatives:
  - shingles/alternatives/loop.py
  - shingles/alternatives/zip_words.py
  - shingles/alternatives/set_methods.py
hints:
  logic: |
    shingles:
      split the text into words
      for each start position where n words still fit:
        join those n words with single spaces
      collect them in a set (a repeat counts once)
        → return the set
    jaccard:
      shared = shingles that are in both a and b
      either = shingles that are in a or b
        → return shared ÷ either
  tools: |
    each start position
      range(n)                  0 up to n − 1
      len(x)                    how many items are in x
    n words from a start position
      seq[a:b]                  items from index a up to b
    join them into one string
      sep.join(items)           glues strings with sep between
    collect without repeats
      {expr for i in seq}       set comprehension, repeats collapse
    shared and either
      a & b                     items in both sets
      a | b                     items in either set
  nudge: "A document of w words has w − n + 1 runs of n words. Jaccard is the size of what two sets share over the size of everything in either."
  concept: "`words[i:i + n]` is the run starting at word i, and `\" \".join(...)` turns it into one string. Start i at 0 and stop at `len(words) - n`, inclusive, so `range(len(words) - n + 1)`. Use a set comprehension so repeats count once. For sets, `a & b` is what they share and `a | b` is everything in either."
  example: |
    Trace it with `n = 3` on `"a b c d e"` and `"a b c d f"`:

    - First text: 5 words, so 5 − 3 + 1 = 3 runs: `{"a b c", "b c d", "c d e"}`.
    - Second text: `{"a b c", "b c d", "c d f"}`.
    - Shared: `"a b c"` and `"b c d"`, so 2. In either: those two plus `"c d e"` and `"c d f"`, so 4.
    - Jaccard is 2 / 4 = `0.5`.

    Edge cases: a text with 2 words has no run of 3, so `shingles("a b")` is the empty set. `"a b a b a"` gives `{"a b a", "b a b"}`, 2 items from 3 runs, because one repeats.
  skeleton: |
    return {" ".join(words[i:i + ___]) for i in range(len(words) - ___ + ___)}
    return len(a ___ b) / len(a ___ b)
  solution: |
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}
    return len(a & b) / len(a | b)
on_solve: "Docs 0 and 1 score 0.82: same story, different ending. Docs 2 and 3 score 0.57, though doc 3 contains all of doc 2 plus a sentence. Jaccard punishes the extra text, which is why some pipelines also check containment. Without step 1, docs 0 and 1 would score 0.63, under the 0.7 threshold you're about to use. Step 4 is unlocked."
mistakes:
  - { file: shingles/mistakes/placeholders.py, fails: shingle_set, says: "`shingles` still returns `...`" }
  - { file: shingles/mistakes/as_list.py, fails: shingle_set, says: "Return a set" }
  - { file: shingles/mistakes/tuples.py, fails: shingle_set, says: "Your shingles are tuples of words" }
  - { file: shingles/mistakes/off_by_one.py, fails: shingle_set, says: "Where does your `range` stop?" }
  - { file: shingles/mistakes/characters.py, fails: shingle_set, says: "runs of 3 *characters*" }
  - { file: shingles/mistakes/fixed_three.py, fails: shingle_set, says: "Use the `n` you're given" }
  - { file: shingles/mistakes/count_only.py, fails: jaccard_value, says: "Jaccard is a share" }
  - { file: shingles/mistakes/overlap.py, fails: jaccard_value, says: "overlap coefficient" }
  - { file: shingles/mistakes/sum_sizes.py, fails: jaccard_value, says: "counts the shared shingles twice" }
  - { file: shingles/mistakes/inverted.py, fails: jaccard_value, says: "Upside down" }
  - { file: shingles/mistakes/distance.py, fails: jaccard_value, says: "Jaccard distance" }
```

### Find near-duplicates with MinHash

```challenge
id: minhash_dedup
file: corpus.py
minutes: 35
start: { from: shingles }
setup:
  - say: "Comparing every pair of shingle sets is fine for seven documents. For a million it's half a trillion comparisons. MinHash squeezes each document into a short signature instead."
    pause: 4.0
  - say: "Here's a seeded hash, and the stub. Each seed puts every possible shingle in a different random order. Write minhash and estimate. The loop at the bottom drops anything that matches a document it already kept."
    type_file: minhash_dedup/stub.py
    pause: 1.2
tests: minhash_dedup/tests.py
solution: minhash_dedup/solution.py
alternatives:
  - minhash_dedup/alternatives/max_hash.py
  - minhash_dedup/alternatives/loop.py
  - minhash_dedup/alternatives/mean_of_bools.py
hints:
  logic: |
    minhash:
      for each seed from 0 to k − 1:
        hash every shingle in the set with that seed
          → this seed's entry = the lowest hash of them all
      → return the k entries as a list
    estimate:
      for each position in the two signatures:
        same number in both?
          yes → count it
      → return count ÷ signature length
  tools: |
    one entry per seed
      [expr for seed in range(k)]   a list, one item per seed
      h(seed, s)                    the seeded hash given above
    the lowest hash
      min(iterable)                 the smallest item
    compare two signatures
      zip(a, b)                     pairs items up by position
      sum(cond for x, y in pairs)   counts the pairs that pass
      x == y                        True when equal
    divide by the length
      len(x)                        how many items are in x
  nudge: "A signature has one entry per seed, not per shingle. Each entry is the smallest hash any of the document's shingles gets under that seed."
  concept: "Loop over seeds on the outside: for seed in range(k), take `min(h(seed, s) for s in shingle_set)`. Two signatures agree at position i exactly when both documents have the same first shingle under seed i, which happens with probability equal to their Jaccard. So the estimate is the number of agreeing positions over k."
  example: |
    Take a document with two shingles, `{"a b c", "b c d"}`, and 3 seeds. The hashes `h(seed, s)` come out as:

    - Seed 0: `"a b c"` → 551424206, `"b c d"` → 3146183771. Smallest: `551424206`.
    - Seed 1: 2080073951 and 1819524790. Smallest: `1819524790`.
    - Seed 2: 1806619566 and 3674739975. Smallest: `1806619566`.
    - The signature is `[551424206, 1819524790, 1806619566]`, three numbers for three seeds.

    For `estimate`, compare `[7, 3, 9, 4]` with `[7, 5, 9, 8]`. Positions 0 and 2 agree, so 2 / 4 = `0.5`. Two identical signatures give `1.0`.
  skeleton: |
    return [___(h(____, s) for s in shingle_set) for seed in range(_)]
    return sum(x __ y for x, y in zip(sig_a, sig_b)) / len(_____)
  solution: |
    return [min(h(seed, s) for s in shingle_set) for seed in range(k)]
    return sum(x == y for x, y in zip(sig_a, sig_b)) / len(sig_a)
on_solve: "The estimate for docs 0 and 1 is 0.83 against an exact 0.82, from 64 numbers per document. The mirror is dropped, 1 of 7, 14%. On a real crawl the same loop runs with 128 seeds and LSH bands, so each new document is compared only with the few that share a band."
mistakes:
  - { file: minhash_dedup/mistakes/placeholders.py, fails: signature, says: "`minhash` still returns `...`" }
  - { file: minhash_dedup/mistakes/per_shingle.py, fails: signature, says: "one per shingle" }
  - { file: minhash_dedup/mistakes/one_seed.py, fails: signature, says: "they all use one seed" }
  - { file: minhash_dedup/mistakes/as_set.py, fails: signature, says: "which loses the order" }
  - { file: minhash_dedup/mistakes/count.py, fails: estimate_value, says: "divide by the signature length" }
  - { file: minhash_dedup/mistakes/set_jaccard.py, fails: estimate_value, says: "Compare position by position" }
  - { file: minhash_dedup/mistakes/disagree.py, fails: estimate_value, says: "*disagree*" }
```

### Run the pipeline on a full crawl

```replay
id: corpus_pipeline
intro: "The same four stages over a full Mongolian crawl, plus exact dedup and the held-out split. It runs on the CPU in about 3 hours, mostly MinHash. Watch a recording first so you know what the report should look like."
log: replays/corpus-pipeline.jsonl
speed: 40
recorded:
  hardware: 16-core CPU, 64 GB RAM
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: Documents left after each stage (thousands)
  flag_above: 1100
  x_labels: [scraped, held-out split]
  caption: "The red bars are the crawl before any filter. MinHash takes the biggest single bite: 21% of what survives the filters is a near-copy of something else."
summary:
  - { label: Documents, value: "1,204k → 700k" }
  - { label: Near-duplicates removed, value: 21.0% }
  - { label: Clean corpus, value: 191M tokens }
  - { label: Held out, value: "2,000 docs, 0.55M tokens" }
run_yourself: |
  # on your machine, from the course repo
  python build_corpus.py \
    --sources data/raw/mn/*.jsonl.gz --out data/mn-clean \
    --shingle 3 --num-perm 128 --threshold 0.7 --heldout 2000
```

### Record your metrics

```metrics
id: record
intro: Your pipeline's near-duplicate share and the clean corpus size. They're compared with the reference run and the pass limits.
```

## Hints

- Run exact dedup (a hash of the normalized text) before MinHash. It's nearly free and removes the easy copies.
- With LSH, the threshold comes from the band shape: b bands of r rows catch pairs above roughly (1/b)^(1/r). 16 bands × 8 rows sits near 0.7.
- Look at what each rule drops. Print 20 random rejects per rule before trusting a threshold. A rule that drops poetry or tables of election results may be too eager for your purpose.
- Keep the source URL and the rule that dropped each document. When 4A.4's samples look odd, you'll want to trace them back.
- Wikipedia is clean but small and formal. News is large but heavily mirrored. Forums are informal and noisy. Write down the mix you end up with. 4B compares runs on it.

## Done when

- [ ] normalize and keep turn twelve scraped documents into seven clean Mongolian ones {#filter auto=challenge:keep_filter}
- [ ] MinHash drops the mirrored weather report and nothing else {#dedup auto=challenge:minhash_dedup}
- [ ] Your clean corpus and its held-out split, made after dedup, are saved {#corpus_saved}
- [ ] Near-duplicate share and corpus size recorded {#metrics_recorded auto=metrics}
