---
gist: A model only learns from the text you feed it. This lesson turns messy scraped Mongolian web pages into a clean corpus by fixing invisible character damage, keeping only real Mongolian, and removing the near-copies the web is full of.
minutes: 17
---

## What this lesson is for {#why}
> Every later run in 4A and 4B trains on the corpus you build here, so its mistakes would end up in every model.

4A.2 proved the training pipeline works. Now it needs the right data. The course project teaches Llama 3.1 8B Mongolian, and the raw material is Mongolian text scraped from the web: news sites, Wikipedia, forums, anything a crawler found.

Raw web text is messy in ways you can't always see. Some of it isn't Mongolian at all, some of it is menus and page counters, and a lot of it is the same news story copied across dozens of sites. If you train on that, the model learns the junk, and your held-out evaluation in 4A.4 ends up grading the model on text it has already seen.

You'll write the four cleaning stages in the browser on twelve sample documents, then watch the same pipeline run over 1.2 million.

## Why Mongolian makes a good test case {#mongolian}
> Mongolian shares an alphabet with Russian, adds two letters of its own, and gets chopped into many small tokens by tokenizers built mostly for other languages.

You don't need to read Mongolian for this lesson. What matters about it:

- **It's written in Cyrillic**, the same alphabet as Russian, plus two extra letters: **ө** and **ү**. Mongolia has used this script since the 1940s.
- **It builds words by adding suffixes.** *хот* is "city", and *хотод* is "in the city". One word carries what English spreads over several.
- **There's comparatively little of it online.** Llama 3.1 saw far more English and Russian than Mongolian.
- **Tokenizers serve it poorly.** A tokenizer learns its pieces from the text it was trained on. Llama 3.1's tokenizer splits English into about 1.3 tokens per word, but the corpus report in this lesson's recording counts about 3.1 tokens per Mongolian word. More tokens per word means more compute per sentence and a harder prediction task.

That combination makes Mongolian a clean test. You can measure whether a model has really learned the language, and 4B can ask whether a tokenizer built for Mongolian helps.

## What you need from before {#before}
> Text is a sequence of Unicode code points, a tokenizer turns it into token ids, and a Python set gives you unions and intersections for free.

- **Code points.** A Python string is a sequence of Unicode code points, the numbers behind each character. `len(s)` counts code points, not what you see on screen. It's the same in Java, where a `String` holds UTF-16 code units.
- **Tokens** (1.5, 1.6). A tokenizer maps text to a list of token ids. Different text, even text that looks identical, can give different tokens.
- **Sets.** `a & b` is what two sets share and `a | b` is everything in either, like Java's `retainAll` and `addAll`, but as expressions.

## Invisible damage: Unicode normalization {#unicode}
> The same letter can arrive as one code point or as two, and Latin letters can pose as Cyrillic ones. Both look perfect on screen and break everything after them.

**Letters in pieces.** Unicode lets you write й either as one code point (U+0439) or as и (U+0438) followed by a *combining breve* (U+0306) that draws the little arc on top. Text pasted from PDFs or old systems often uses the two-piece form. On screen they're identical. To Python, a tokenizer or a hash, they're different strings:

| Written as | Code points | `len` | Equal to the other? |
|---|---|---|---|
| й, typed | U+0439 | 1 | no |
| и + breve, pasted | U+0438 U+0306 | 2 | no |

**NFC normalization** puts pieces back together wherever a single code point exists. Its opposite, **NFD**, takes letters apart. That's one of the classic mistakes the tests catch.

**Look-alikes.** Latin `a`, `o`, `e`, `c`, `p`, `x` look exactly like Cyrillic `а`, `о`, `е`, `с`, `р`, `х`. Someone typing on the wrong keyboard layout produces words like *Улаанбаатaр* with a Latin `a` inside. The lesson gives you a table of look-alikes. The rule is to replace them **only inside words that already contain Cyrillic**, so a quoted English brand name like *Facebook* keeps its Latin letters.

Order matters. Compose first, then fix look-alikes. A Latin `o` followed by a combining diaeresis should become `ö` and then Cyrillic `ө`. If you swap the `o` first, composing produces a different Cyrillic letter, `ӧ`, which Mongolian doesn't use.

Finally, collapse every run of whitespace (tabs, non-breaking spaces, double spaces) into one plain space.

## Is it Mongolian? Letters as a fingerprint {#language}
> A Cyrillic test can't tell Mongolian from Russian or Kazakh. Mongolian's ө and ү rule out Russian, and Kazakh's extra letters rule out Kazakh.

Three languages crowd Mongolian on the web, two of them in the same alphabet:

| Language | Script | Has ө or ү? | Has ә, ғ, қ, ң, ұ, һ, і? |
|---|---|---|---|
| Mongolian | Cyrillic | yes | no |
| Russian | Cyrillic | no | no |
| Kazakh | Cyrillic | yes | yes |
| English | Latin | no | no |

So the filter has two letter checks. First, most of the **letters** must be Cyrillic: at least 80% of them. That catches English. Then the text must contain ө or ү (that catches Russian) and none of Kazakh's own letters (that catches Kazakh).

The tests watch two traps:

- **Divide by letters, not by every character.** A sentence like "2024 оны 7 сарын 11-нд …" is all Cyrillic letters, but digits and spaces make it look only about 70% Cyrillic if you count every character.
- **Ө *or* ү, not both.** Plenty of Mongolian sentences use only one of them.

A rule that drops a sentence for containing a single Latin word, like a quoted brand name, is too strict. 80% of the letters is a share, not all-or-nothing.

## Quality: too short, or not really text {#quality}
> Very short fragments and pages of digits and symbols teach the model nothing useful, so they go before anything else.

Two cheap rules come first:

- **At least 6 words.** Footers like "All rights reserved", menu items and captions are shorter. Six exactly is fine: "fewer than 6" means 5 or less.
- **Letters are at least half of the characters that aren't spaces.** A page counter like "1. 2. 3. … Үзсэн: 1520" has a few Cyrillic words, but mostly digits and symbols.

On the twelve samples, each rule catches exactly one kind of junk, and seven clean Mongolian documents survive.

## Exact and near duplicates {#duplicates}
> Exact duplicates are easy to remove with a hash. Near-duplicates, the same story with a different footer, need a similarity measure.

News is copied constantly. One agency writes a story and twenty sites republish it, each adding its own header, footer or source line. Two problems follow if you keep the copies. The model sees that story twenty times as often as anything else, so it memorises it. And when you split off held-out text later, copies end up on both sides, so the model is graded on text it has seen.

**Exact** duplicates are easy: hash the normalized text and keep one document per hash. It's the same idea as using a `HashSet<String>` in Java. **Near**-duplicates differ by a few words, so every hash is different. They need a way to say "these two are 80% the same".

## Shingles and Jaccard similarity {#jaccard}
> Cut each document into overlapping runs of 3 words, then compare the two sets: what they share divided by everything in either.

Use letters as stand-in words. Document 1 is `A B C D E` and document 2 is `A B C D E F`. Their 3-word **shingles** are:

```
doc 1: ABC  BCD  CDE
doc 2: ABC  BCD  CDE  DEF
```

A document of w words has w − 3 + 1 shingles. Forgetting the "+ 1" loses the last one, a mistake the tests catch.

**Jaccard similarity** is the size of the intersection over the size of the union:

```
shared:   ABC BCD CDE      → 3
either:   ABC BCD CDE DEF  → 4
Jaccard = 3 / 4 = 0.75
```

Identical documents score 1 and documents with nothing in common score 0. The lesson treats anything at 0.7 or above as a near-duplicate. The tests catch the usual slips: dividing by the smaller set (that's a different measure, the overlap coefficient), dividing by the two sizes added together (that counts the shared shingles twice), or returning the count instead of the share.

Normalization pays off here. The mirrored weather report in the samples scores 0.82 against its original. Without the look-alike fix, its first two words would differ, and it would score 0.63, under the threshold.

## MinHash: comparing without the full sets {#minhash}
> Shuffle every possible shingle into a random order and see which of a document's shingles comes first. Two documents pick the same one with probability equal to their Jaccard.

Comparing shingle sets pairwise is fine for seven documents. For a million it's about half a trillion pairs. **MinHash** squeezes each document into a short list of numbers instead.

Imagine every possible shingle in the world written on a card, and the whole deck shuffled. Now look at document 1's shingles and ask which one comes first in the deck. Do the same for document 2. The first card of their *combined* shingles is equally likely to be any of them. Both documents pick that same card only when it's one they share. So:

```
chance both pick the same card
  = shared / either
  = Jaccard
```

Repeat with 64 different shuffles and count how often they agree. If they agree 53 times out of 64, the estimate is 0.83.

A computer doesn't shuffle decks. A **seeded hash function** does the job: hash seed 0 gives every shingle a random-looking number, and the shingle with the smallest number "comes first". Seed 1 gives a different order, and so on. Each document's **signature** is its 64 minimums, one per seed, in order. The estimate is the share of positions where two signatures agree.

Two classic mistakes: making one entry per *shingle* instead of one per *seed*, and comparing signatures as sets instead of position by position. Position i means "seed i's winner", so positions only mean something side by side.

At scale, a trick called **locality-sensitive hashing (LSH)** groups signatures into buckets, so each document is only compared with the few that already share a bucket.

## The full pipeline and the held-out split {#pipeline}
> Normalize, filter, remove exact copies, remove near-copies, and only then split off the held-out set and count tokens.

The order matters. Normalizing first means the filters and hashes see clean text. Cheap filters run before expensive dedup. And the **held-out split**, the documents 4A.4 measures perplexity on, comes *after* dedup, so no held-out document has a near-copy in training.

On your machine the pipeline runs on the CPU in about 3 hours, mostly MinHash. The replay is an **illustrative** recording written for the course, not a real run. A healthy report looks like this:

| Stage | Documents left |
|---|---|
| scraped | 1,204k |
| after language filter | 1,079k |
| after quality filter | 998k |
| after exact dedup | 886k |
| after near dedup (21.0% removed) | 700k |

The clean corpus comes to about 191 million tokens with Llama 3.1's tokenizer, and 2,000 documents are held out. Read a sample of what each rule rejects before you trust a threshold.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Corpus**: the collection of text a model trains on.
- **Code point**: the number Unicode assigns to one character, written like U+0439.
- **Combining character**: a code point that modifies the one before it, like the breve in и + ◌̆.
- **NFC / NFD / NFKC**: Unicode normal forms. NFC composes letters into single code points, NFD decomposes them, NFKC also folds compatibility characters.
- **Look-alike (homoglyph)**: a character from another script that looks identical, like Latin `a` and Cyrillic `а`.
- **Language ID**: deciding which language a text is in. Here it's done with letter rules.
- **Kazakh-only letters**: ә, ғ, қ, ң, ұ, һ, і. Kazakh uses them and Mongolian doesn't.
- **Boilerplate**: repeated page furniture like menus, footers and "All rights reserved".
- **Exact dedup**: removing documents whose normalized text is identical.
- **Near-duplicate**: a document that is almost the same as another, often a mirror with a different header or footer.
- **Shingle**: an overlapping run of n consecutive words (n = 3 here).
- **Jaccard similarity**: the size of the intersection over the size of the union of two sets.
- **Containment**: how much of one document's shingle set appears in another's. It catches a short document copied inside a long one.
- **MinHash**: estimating Jaccard from the minimum hash values of each set under many seeds.
- **Signature**: a document's list of k minimums, one per seed.
- **Seed**: a number that selects one random-looking hash function, and so one shuffled order.
- **LSH (locality-sensitive hashing)**: bucketing signatures so only likely matches get compared.
- **Threshold**: the similarity above which two documents count as near-duplicates, 0.7 here.
- **Held-out split**: documents kept out of training and used only for evaluation.
- **Near-duplicates removed (%)**: the lesson's quality metric, the share of documents MinHash removes.
- **Corpus size (M tokens)**: the lesson's systems metric, the clean corpus counted with Llama 3.1's tokenizer.
