---
gist: A BPE tokenizer starts from raw bytes and repeatedly glues the most common neighbouring pair into a new token, so frequent pieces of text become single tokens and nothing is ever unknown.
minutes: 16
---

## What this lesson is for {#why}
> Models read token ids, not text. This lesson builds the algorithm that turns text into ids and back, the same family Llama 3.1 uses.

Every model so far read characters: makemore read 27 letters, the Shakespeare models read 65 characters. Real language models read **tokens**: pieces of text that are often whole words or common word parts. 1.4's TinyStories model uses a 21,504-token vocabulary, and Llama 3.1, which phase 4 fine-tunes, uses 128,256.

The algorithm behind almost all of them is **byte-pair encoding (BPE)**. It's short enough to write in an afternoon: count neighbouring pairs, merge the most common one, repeat. You'll train it on a short story, then use it to encode and decode text in five scripts, including Mongolian. 1.6 then measures how badly a tokenizer can treat a language it didn't see much of, and phase 4B trains a Mongolian one.

## What you need from before {#before}
> Dictionaries, zip and bigram counting, all from phase 0.

- **Dicts**: `d.get(key, 0)` returns the value for `key`, or 0 if the key isn't there yet, like Java's `getOrDefault`. That's the usual way to keep counts in a dict without checking first.
- **zip with a shifted copy**: `zip(xs, xs[1:])` gives every pair of neighbours. In 0.0 that was `("e", "m"), ("m", "m"), …` for "emma".
- **Bigram counts**: makemore counted how often each character followed each other one. BPE starts with exactly that count.

This whole lesson is plain Python, no tensors.

## Text is bytes {#bytes}
> UTF-8 stores every character as 1 to 4 bytes, and each byte is a number from 0 to 255. BPE starts from these numbers.

Python keeps text (`str`) and raw bytes (`bytes`) apart. `encode` goes from text to bytes and `decode` goes back, like `getBytes(UTF_8)` and `new String(bytes, UTF_8)` in Java:

```python
"hi".encode("utf-8")
# b'hi'

list("hi".encode("utf-8"))
# [104, 105]

list("ө".encode("utf-8"))
# [211, 169]
```

A `b'...'` literal is a bytes value. Different scripts take different numbers of bytes per character:

| Script | Example | Bytes per character |
|---|---|---|
| English (ASCII) | `a` | 1 |
| Cyrillic | `а`, `ө` | 2 |
| Chinese | `你` | 3 |
| Emoji | `🙂` | 4 |

So "Сайн байна уу?" is 14 characters but 25 bytes: 11 Cyrillic letters at 2 bytes, plus 2 spaces and a question mark at 1 byte each.

## Why not characters, or words? {#why-subwords}
> Words make the vocabulary huge and leave unknown words; characters make sequences long. Bytes plus merges sit in between.

| Unit | Vocabulary | Problem |
|---|---|---|
| whole words | hundreds of thousands | new names and typos are unknown |
| characters | ~150,000 in Unicode | sequences are long |
| bytes | 256 | sequences are even longer |
| bytes + merges | your choice | none of the above |

Starting from bytes means the base vocabulary is just 256 symbols and **any** text, in any script, can be encoded. Merges then shorten the text: every merge turns a frequent pair of tokens into one new token. The number of merges you do sets the vocabulary size: 256 + merges.

## Counting pairs {#pairs}
> Walk the list of ids and count every pair of neighbours. Neighbours overlap: the middle id of three belongs to two pairs.

Take `"aaabdaaabac"`, where a = 97, b = 98, c = 99 and d = 100:

| Pair | Letters | Count |
|---|---|---|
| (97, 97) | aa | 4 |
| (97, 98) | ab | 2 |
| (98, 100) | bd | 1 |
| (100, 97) | da | 1 |
| (98, 97) | ba | 1 |
| (97, 99) | ac | 1 |

"aaa" contains "aa" twice, at positions 0–1 and 1–2, which is how "aa" reaches 4. Counting only pairs that start at even positions misses half of them, and a test checks that. The other edge to watch is the end of the list: the last id has no neighbour after it, so a list of n ids has n − 1 pairs.

## Merging a pair {#merge}
> Copy the list left to right, writing the new id wherever the pair starts, and skip over both of its ids. Merges never overlap.

Merging "aa" (97, 97) into the new id 256:

```
before
[97,97,97,98,100,97,97,97,98,97,99]

after
[256,97,98,100,256,97,98,97,99]
```

In "aaab", the first two a's merge. The third a has lost its partner, so it stays. Three things go wrong in practice:

| Mistake | Result on [1, 2, 3, 1, 2], pair (1, 2) |
|---|---|
| step forward 1 after a merge | `[99, 2, 3, 99, 2]` |
| check only the first id | also merges (1, 3) elsewhere |
| look at ids[i + 1] at the last i | IndexError |

The correct result is `[99, 3, 99]`.

## Training a vocabulary {#train}
> Repeat: count pairs, take the most frequent, give it the next id (256, 257, …), merge it everywhere, and record which bytes the new token stands for.

Two dicts hold the result. **merges** maps a pair to its new id, in the order they were learned. **vocab** maps every id to the bytes it stands for: ids 0–255 stand for one byte each, and a merged token stands for its first part's bytes followed by its second part's. That works even when a part is itself a merged token.

Picking "the most frequent pair" needs care in Python. `max(some_dict)` compares the dict's **keys**, so on pair counts it returns the pair that sorts last, not the most common one. `max` takes a `key=` argument that says what to compare by, as in `max(["pear", "fig"], key=len)`, which returns "pear". Comparing the pairs by their counts is the same idea.

On this lesson's 792-byte story, the first merges are ' t', 'e ', ' th', 'd ', 't ', and then the sixth is ' the ': the algorithm discovers English's most common word by itself. 20 merges shrink the story from 792 tokens to 548.

## Encoding new text {#encode}
> To encode text, start from its bytes and apply the learned merges in the order training learned them, earliest first, until none apply.

Later merges were built on top of earlier ones: ' th' only exists because ' t' was merged first. So encoding replays the merges in training order. At each round, look at the pairs present in the text, pick the one with the **smallest** merge id, merge it, and repeat. When no pair in the text was ever merged, stop.

Why not just merge the most frequent pair in the new text? Take merges learned in this order: "bc" → 256, then "ab" → 257. Encode "abcab":

| Strategy | Result |
|---|---|
| training order | `[97, 256, 257]` (a, bc, ab) |
| most frequent first | `[257, 99, 257]` (ab, c, ab) |

The second one is a split training never produced, so the same text could get different tokens in different contexts. Most pairs in a new text were never merged at all, so they need a rank that can never win; the stop test handles the end.

## Decoding {#decode}
> Look up each id's bytes, join all of them into one bytes value, and decode that once.

A token can hold half a character. Cyrillic letters are 2 bytes, and a tokenizer may have merged "a" with the first byte of "С" into one token. Decoding each token on its own would fail on the half-letters. Joining every token's bytes first puts each character back together.

Joining needs a bytes separator. `"".join(...)` joins strings, and on bytes it raises "expected str instance, bytes found"; the bytes version starts with `b""`. And `bytes(ids)` only works for ids up to 255, because a merged id like 300 isn't a byte.

Decoding uses `errors="replace"`. Text that was encoded properly never needs it, but a model generating one token at a time can stop in the middle of a character, and without it decoding would crash. A broken piece becomes `�`.

## Round trip and speed {#metrics}
> Your quality metric is the share of test lines that come back exactly after encode then decode. The systems metric is encoding speed in MB/s.

The test set has eight lines: English, Mongolian Cyrillic, accented Latin, Chinese, emoji and Python code. Your tokenizer was trained on one English story, yet all eight should round-trip, because every text is bytes. The pass mark is 100%.

They don't all compress, though. "Сайн байна уу?" comes out as 25 tokens for 14 characters: the story has no Cyrillic, so no merge applies and every byte is a token. That's the problem 1.6 measures on a real tokenizer.

Speed is about 0.09 MB/s in the browser. Your encoder recounts every pair on every round, in plain Python. Production tokenizers like tiktoken keep smarter data structures and are written in Rust, and they're thousands of times faster.

## Words you'll see {#glossary}
> The terms in this lesson's cards, challenges and metrics.

- **Token**: one unit a model reads, from a single byte to a whole word.
- **Tokenizer**: the code that turns text into token ids and back.
- **BPE (byte-pair encoding)**: build a vocabulary by repeatedly merging the most frequent neighbouring pair.
- **Byte-level BPE**: BPE that starts from the 256 byte values, so no text is ever unknown.
- **UTF-8**: the standard way to store text as bytes, 1 to 4 per character.
- **bytes / str**: Python's types for raw bytes and for text. `encode` and `decode` convert.
- **Pair**: two neighbouring ids, like (97, 97).
- **Merge**: replacing every occurrence of a pair with one new id.
- **merges**: the dict of pair → new id, in training order.
- **vocab**: the dict of id → the bytes it stands for.
- **Vocabulary size**: 256 + the number of merges.
- **Encode / decode**: text → ids, and ids → text.
- **Round trip**: decode(encode(text)) gives back exactly the text.
- **`errors="replace"`**: decode broken bytes as `�` instead of raising.
- **MB/s**: megabytes of text encoded per second.
