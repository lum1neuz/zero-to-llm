---
gist: Llama 3.1 chops Mongolian words into too many small pieces, which makes every Mongolian sentence slower and more expensive. This lesson measures that cost, trains a tokenizer that fits Mongolian, and picks which of its tokens are worth adding.
minutes: 16
---

## What this lesson is for {#why}
> Phase 4B asks one research question: does giving Llama 3.1 Mongolian-sized tokens make it a better and cheaper Mongolian model? This lesson builds those tokens.

In 4A you taught Llama 3.1 8B more Mongolian without touching its tokenizer: the **control run** in 4A.4, then instruction tuning in 4A.5. Phase 4B repeats that work with one change, a **bigger vocabulary** that includes Mongolian tokens, and compares the results fairly in 4B.4.

The four lessons are one pipeline:

| Lesson | What you do |
|---|---|
| 4B.1 | Choose 8,000 new Mongolian tokens |
| 4B.2 | Give the model a row for each one |
| 4B.3 | Train those rows on Mongolian text |
| 4B.4 | Compare four models on one eval |

This lesson is the first step, and it mostly runs on a CPU. In the browser you measure the problem and write the selection logic on a tiny corpus. On your machine you do the same on the 4A.3 corpus.

## What you need from before {#before}
> A tokenizer turns text into ids from a fixed list called the vocabulary. You built a BPE tokenizer in 1.5 and measured Llama 3.1's on Mongolian in 1.6.

A quick refresher:

- **Token**: one entry of the vocabulary. It can be a whole word, part of a word or a single byte.
- **Vocabulary**: the list of all tokens. Llama 3.1 has 128,256. Each token's position in the list is its **id**.
- **BPE** (byte pair encoding): the training method from 1.5. Start from single characters and keep merging the most frequent neighbouring pair into a new token.
- **The audit in 1.6**: you counted how many tokens Llama 3.1 spends per Mongolian word. In the course's reference run that's **3.38**, against **1.31** for English. Your own number may differ a little. Use yours.

## A few words of Mongolian {#mongolian}
> Mongolian is written in Cyrillic and builds words by adding suffixes to a stem, so one stem appears in many slightly different words.

Mongolian uses the Russian alphabet plus two extra letters, **ө** and **ү**. You don't need to read it, but the lesson's examples will make more sense if you see how words are built:

| Word | Meaning | Built from |
|---|---|---|
| хүүхэд | child | stem |
| хүүхдүүд | children | stem + -үүд (plural) |
| сургууль | school | stem |
| сургуульд | at school | stem + -д (at/in) |
| сургуульдаа | to one's own school | stem + -д + -аа (one's own) |
| амьдардаг | lives (usually) | амьдар- (live) + -даг (habitually) |
| байна | is, is doing | a very common helper verb |

A language like this is called **agglutinative**: grammar is glued onto words as suffixes. That's bad news for a tokenizer trained mostly on English. It has never seen these stems and suffixes often enough to merge them, so it spells them out in short pieces.

## Fertility: tokens per word {#fertility}
> Fertility is total tokens divided by total words. It tells you how many tokens a language costs, and lower is cheaper.

Count the words by splitting on whitespace. Encode each word and add up the tokens. Divide.

```
sentence (5 words):
Хүүхдүүд өглөө бүр
сургуульдаа явдаг.
"The children go to school
every morning."

Llama 3.1:  17 tokens
fertility:  17 / 5 = 3.4
```

Three classic slips, all of which the challenge tests check for:

- **Counting letters instead of tokens.** Letters per word is not what a model pays for.
- **Dividing the wrong way round.** Words per token comes out below 1. Fertility is tokens per word.
- **Counting each different token once.** If "байна" costs two tokens and appears three times, that's six tokens, not two. Repeats count.

Split words the same way every time. In the lesson, `words()` lowercases and drops punctuation. If you split with `text.split()` instead, "байна." keeps its full stop and costs an extra token, so the number changes for a reason that has nothing to do with the tokenizer.

## Why fertility costs money and context {#cost}
> Every token is one step of generation, one slot in the context window and one unit on a per-token bill. More tokens per word makes Mongolian slower, shorter-sighted and more expensive.

A model generates one token per step. A context window holds a fixed number of tokens, not words. So at a fertility of 3.38:

| | Llama 3.1 | Extended (1.62) |
|---|---|---|
| Words in 8,192 tokens | 8,192 / 3.38 ≈ 2,420 | 8,192 / 1.62 ≈ 5,060 |
| Tokens for a 100-word answer | 338 | 162 |
| Relative cost per word | 1.0× | about 0.48× |

The same 100-word answer takes 338 decode steps instead of 162. A hosted model that charges per token charges more than twice as much. And the model has a harder job: it has to get a word right across three or four predictions instead of one or two.

## Training a tokenizer: BPE and unigram {#training}
> BPE builds a vocabulary up from characters by merging frequent pairs. Unigram starts from many candidate pieces and prunes away the least useful ones. Both learn which pieces of text are common enough to deserve a token.

**BPE**, which you know from 1.5, works bottom-up. Count every neighbouring pair in the training words, weighted by how often each word occurs. Merge the most frequent pair everywhere. Repeat until the vocabulary is the size you want:

```
start:   с у р г у у л ь
merge 1: с у р г уу л ь
merge 2: с у р г уу ль
...
later:   сургууль
```

Each merge adds one token. After enough merges, frequent words like **сургууль** (school) become single tokens and rare words stay in pieces.

**Unigram** (used by SentencePiece, for example) works top-down. Start with a large set of candidate pieces and give each a probability. Repeatedly drop the pieces whose removal hurts the overall likelihood of the training text least. To encode a word, it picks the most probable way to split it.

For this course the difference doesn't matter much. Both produce a vocabulary of pieces that fit the language they were trained on. The lesson uses BPE because you already wrote one, and Llama 3.1's own tokenizer is a byte-level BPE, so the new pieces line up with how Llama cuts text.

## Measure on text the trainer never saw {#held-out}
> Fertility on the training text flatters the tokenizer. Always measure on held-out text.

A tokenizer trained on a sentence has merges for that sentence's exact words. On a small corpus BPE learns whole training words, so the training text tokenizes beautifully. What matters is how it does on text it hasn't seen, which is what your users will type.

The lesson keeps a few **held-out** sentences out of training and measures on those. The real run uses 4A.3's held-out split of 2,000 documents. Split by document, not by sentence. Two near-identical news articles on either side of the split leak words across it.

## Greedy longest match {#encode}
> The browser lessons encode a word by taking the longest vocabulary piece that fits at each position. It's a simple stand-in for real BPE encoding.

Real BPE encoding replays the learned merges in order. The lesson uses a simpler rule that gives very similar counts and lets you add tokens by just adding strings to a set:

```
vocab: б а й н ай на бай

байна → try "байна": no
        try "байн":  no
        try "бай":   yes
then    "на":        yes
result: бай | на  (2 tokens)
```

Start at the first letter, try the longest string first, take the first match and move on. A single letter always matches, so nothing is ever unknown.

## Which tokens are worth adding {#choosing}
> A new token is worth what it saves: how often it appears, times the base tokens it replaces minus the one it costs.

The Mongolian BPE has thousands of tokens that Llama lacks. You won't add all of them, because every new token is a new row of numbers to train in 4B.2 and 4B.3. Rank them by tokens saved:

```
saved = count × (pieces − 1)
```

Here **count** is how often the token appears in the corpus, and **pieces** is how many tokens Llama's vocabulary spends on it today. The minus one is the new token itself, which still costs one token.

| Token | Meaning | Count | Llama pieces | Saved |
|---|---|---|---|---|
| ын | of (suffix) | 80,000 | 2 | 80,000 |
| байгаа | being, that is | 25,000 | 5 | 100,000 |

Ranked by count alone, "ын" wins. Ranked by savings, "байгаа" wins, because each use turns five pieces into one.

Mistakes the tests check for: ranking by raw count (which also lets in tokens Llama already has, saving nothing), scoring by letters instead of Llama's pieces, forgetting the minus one, and sorting smallest first.

## How many to add {#how-many}
> The first few thousand tokens do most of the work. After that each extra token saves little and still costs a row to train.

The real run trains a 16,000-token Mongolian BPE. 2,914 of its tokens are already in Llama's vocabulary, which leaves 13,086 candidates. Adding them in rank order gives held-out fertility like this (reference run, illustrative):

| Tokens added | Fertility |
|---|---|
| 0 | 3.38 |
| 1,000 | 2.21 |
| 2,000 | 1.98 |
| 4,000 | 1.78 |
| 8,000 | 1.62 |
| 12,000 | 1.57 |

The curve flattens. Token 12,000 might appear a few hundred times in the whole corpus, too rarely to learn a good embedding. The course adds 8,000, which makes the vocabulary 128,256 + 8,000 = 136,256, a multiple of 64. GPUs multiply matrices faster when sizes line up like that.

## What you'll run {#run}
> On your machine: train a 16k BPE on 4A.3's train split, rank its tokens by savings, add the top 8,000, and check held-out fertility. The recording in the lesson is illustrative sample data.

It's a CPU job of about ten minutes. The replay in the lesson shows what healthy output looks like: fertility falling from 3.38 to about 1.62, a round-trip check that decoding every encoded held-out document gives back the original text, and a saved tokenizer with `new_tokens.txt` in rank order. The numbers in it are written for the course, not recorded, so expect yours to differ a little.

You record two metrics: your held-out fertility (pass: under 2.0) and the number of tokens you added (pass: at most 16,000).

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Token**: one entry of a tokenizer's vocabulary: a word, part of a word or a byte.
- **Vocabulary (vocab)**: the full list of tokens. Llama 3.1's has 128,256.
- **Fertility**: tokens per word, total tokens divided by whitespace words. Lower is cheaper.
- **BPE (byte pair encoding)**: a tokenizer trained by repeatedly merging the most frequent neighbouring pair.
- **Merge**: one BPE training step. It adds one token to the vocabulary.
- **Unigram tokenizer**: a tokenizer trained by pruning a large set of candidate pieces down to the most useful ones.
- **Byte-level BPE**: BPE over raw bytes, as Llama 3.1 uses, so any text can be encoded.
- **Greedy longest match**: encoding by taking the longest vocabulary piece at each position. The browser's stand-in for BPE encoding.
- **Held-out text**: text kept out of training, used only to measure.
- **Candidate**: a Mongolian token that Llama's vocabulary doesn't have yet.
- **Tokens saved**: count × (base pieces − 1), the score used to rank candidates.
- **Extended vocabulary**: Llama's 128,256 tokens plus your new ones, 136,256 in the reference run.
- **Agglutinative**: a language that builds words by adding suffixes to stems, like Mongolian.
