---
gist: A tokenizer that saw little Mongolian splits it into many small pieces, sometimes single bytes, and three numbers measure how bad that is before you decide how to fix it.
minutes: 14
---

## What this lesson is for {#why}
> Phase 4 teaches Llama 3.1 Mongolian. Before that, you measure how Llama's tokenizer cuts Mongolian text, because every extra token costs context, compute and money.

In 1.5 your tokenizer trained on an English story still encoded Mongolian perfectly, but at a cost: 25 tokens for 14 characters, because no merge ever applied. Real tokenizers have the same problem in a milder form. A language that was rare in the tokenizer's training data gets few merges, so its words are split into many small pieces.

This lesson is an **audit**: three measurements you can run on any tokenizer. You write them in the browser and try them on two small tokenizers made with 1.5's code, then run the same audit on Llama 3.1's real tokenizer on your own machine. The result decides a big question in phase 4: keep Llama's tokenizer as it is (4A), or train a Mongolian one and extend the vocabulary (4B).

## What you need from before {#before}
> Bytes, merges and encode from 1.5.

- **UTF-8**: every character is 1 to 4 bytes. Cyrillic letters are 2.
- **Byte-level BPE**: start from the 256 byte values and merge frequent pairs into new tokens.
- **encode(text, merges)**: replay the merges on the text's bytes and return the token ids.
- **vocab**: maps each id to the bytes it stands for. Ids 0–255 are single bytes; merged tokens start at 256.

The browser lesson gives you 1.5's tokenizer ready-made, so you can focus on measuring.

## Mongolian in Unicode {#cyrillic}
> Mongolian is written in the Cyrillic alphabet: Russian's 33 letters plus Ө, ө, Ү and ү. All of them live in Unicode's Cyrillic block, U+0400 to U+04FF.

Every character has a number called its **code point**, written U+ and four hex digits. Python's `ord("a")` gives it as a plain integer, 97, and `ord("ж")` gives 1078. The **Cyrillic block** is the range U+0400 to U+04FF (1024 to 1279), and it contains every Cyrillic letter in both cases.

The trap is assuming Cyrillic means а to я:

| Letter | Code point | Inside а…я? |
|---|---|---|
| а | U+0430 | yes |
| я | U+044F | yes |
| ё | U+0451 | no |
| ү | U+04AF | no |
| ө | U+04E9 | no |
| Ж | U+0416 | no (capital) |

а…я is Russian's lowercase run. Mongolian's own letters sit later in the block, so a check written for Russian silently misses them. The opposite trap is checking "not ASCII", which counts Greek and Chinese as Cyrillic. Step 3's tests try both.

In UTF-8, ө is the two bytes `[211, 169]`. Its code point is 1257, but no byte can be bigger than 255, which is why one letter takes two.

## Fertility {#fertility}
> Fertility is tokens per word: the total number of tokens a sample produces, divided by its total number of words.

Words are counted the simple way, by splitting on spaces. For a sample of sentences you add up all the tokens and all the words, then divide once:

```
sentence 1: 3 tokens, 2 words
sentence 2: 1 token,  1 word

fertility = (3 + 1) / (2 + 1)
          = 4 / 3 ≈ 1.33
```

Averaging each sentence's own ratio instead would give (1.5 + 1) / 2 = 1.25, which lets a short sentence count as much as a long one. The tests also check that you count *tokens*, not characters or bytes, and that your function uses the tokenizer it's given rather than one from elsewhere in the script.

A good tokenizer gives English about 1.3 tokens per word: most common words are one token. A language the tokenizer barely knows can need several times that.

## Byte pieces {#pieces}
> When a letter has no merge, a byte-level tokenizer falls back to its raw bytes. Each byte alone is half a letter, and it can't be decoded by itself.

The browser's Mongolian tokenizer, trained on a few paragraphs, never merged the capital Ө. Here's how it encodes Өнөөдөр ("today"), with each token decoded on its own:

```
7 letters:
Ө н ө ө д ө р

8 tokens:
['�','�','н','ө','ө','д','ө','р']
```

The capital Ө came out as two tokens, each half a letter, shown as `�`. The model has to learn that those two bytes together mean Ө, from context, extra work that a merged letter doesn't need. That's what **byte fallback** means here.

You can tell whether a token is a whole character or a piece by trying to decode its bytes. Python raises `UnicodeDecodeError` on bytes that stop in the middle of a character. Decoding with `errors="replace"` would never raise, so it can't tell you anything, which is one of the mistakes step 2 checks for. Length doesn't tell you either: `b'a'` is one byte and a whole letter, and `b'\xd0\xb0'` is two bytes and also a whole letter, а.

## Cyrillic coverage {#coverage}
> Of all the merged tokens in a vocabulary, what share contain at least one Cyrillic letter? It shows how much of the vocabulary was spent on Cyrillic at all.

Only merged tokens count, ids 256 and up. The 256 single bytes are the same in every byte-level tokenizer and none of them is a Cyrillic letter, so including them only dilutes the share. Some merged tokens are themselves half a letter, so decode with `errors="replace"`: the piece becomes `�`, which isn't Cyrillic, and nothing crashes.

## The two browser tokenizers {#browser}
> A tokenizer that never saw Mongolian spends one token per byte. 40 merges learned from a few Mongolian paragraphs nearly halve that.

Both tokenizers in the browser come from 1.5's code with 40 merges each. One was trained on an English story, the other on a few paragraphs of Mongolian. Measured on six held-out Mongolian sentences:

| Measure | Bytes only | English BPE | Mongolian BPE |
|---|---|---|---|
| tokens per word | 11.5 | 11.5 | 6.0 |
| tokens that are byte pieces | – | 90.7% | 24.8% |
| merged tokens with Cyrillic | – | 0% | 95% |

These numbers are real: the browser computes them as you run each step. The English tokenizer is no better than raw bytes on Mongolian, because none of its merges ever match Cyrillic bytes.

## What fertility costs {#cost}
> Everything in an LLM is counted in tokens, so higher fertility means more context used, more compute, higher prices and slower generation for the same text.

Take an 8,192-token context. At 1.3 tokens per word it holds about 6,300 English words. At 3.4 tokens per word it holds about 2,410 Mongolian words, under half. Each generated token is a full forward pass, so a Mongolian answer of the same length also takes about 2.6 times as many steps to write. And during training the model sees fewer Mongolian words per token of compute.

## Auditing Llama 3.1 {#llama}
> On your own machine, load Llama 3.1's tokenizer with the transformers library and run the same three measures on a larger Mongolian sample and its English translation.

The recording shows the shape of the output: the vocabulary size, the share of merged tokens with Cyrillic, fertility for English and Mongolian, the share of byte pieces, and a few example splits. It's **illustrative**: sample output written for the course, and its numbers (3.38 tokens per Mongolian word against 1.31 for English, 4.6% Cyrillic tokens) are not measured Llama 3.1 values. Your run gives the real ones, and those are your metrics.

A few practical notes:

- The tokenizer download needs a Hugging Face account with access to Llama 3.1, and runs fine on a CPU.
- Compare the **same** sentences in both languages, so the ratio measures the tokenizer, not the content.
- `tokenizer.convert_ids_to_tokens` shows tokens through GPT-2's byte-to-character mapping, so Cyrillic looks like `Ð°`. Decode each id with `tokenizer.decode([i])` to see real text.
- Save ten common Mongolian words and how they split. In 4B.1 you train a Mongolian tokenizer and compare against this list.

## Words you'll see {#glossary}
> The terms in this lesson's cards, challenges and metrics.

- **Audit**: a set of measurements of how a tokenizer treats some text.
- **Fertility**: total tokens divided by total words, over a sample.
- **Code point**: a character's Unicode number, like U+04E9 for ө. `ord()` returns it.
- **Cyrillic block**: Unicode's range U+0400 to U+04FF, every Cyrillic letter in both cases.
- **Ө, ө, Ү, ү**: the letters Mongolian adds to the Russian alphabet.
- **Byte piece / partial token**: a token whose bytes are only part of a character.
- **Byte fallback**: a byte-level tokenizer encoding an unmerged character as its raw bytes.
- **UnicodeDecodeError**: what Python raises when bytes don't form whole characters.
- **`errors="replace"`**: decode broken bytes as `�` instead of raising.
- **Merged token**: a token with id 256 or more, made by a merge.
- **Cyrillic coverage**: the share of merged tokens that contain a Cyrillic letter.
- **Held-out sentences**: test text the tokenizer wasn't trained on.
- **Vocab extension**: adding new tokens to an existing tokenizer and model (4B.2).
- **transformers**: the Hugging Face library that loads Llama 3.1's tokenizer.
