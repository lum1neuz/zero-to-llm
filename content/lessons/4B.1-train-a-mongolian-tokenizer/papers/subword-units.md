---
paper: Neural Machine Translation of Rare Words with Subword Units
gist: Split rare words into frequent subword pieces learned with byte pair encoding, and a fixed-size vocabulary can still represent any word.
---

## The problem {#problem}
> Translation models had a fixed word vocabulary, so rare words, names and compounds became an "unknown" token.

Neural translation systems of the time kept a vocabulary of tens of thousands of whole words. Everything else was mapped to one unknown symbol and patched up afterwards with dictionaries or copying tricks. Languages that build long words out of parts, like German compounds, suffer most, because so many of their words are rare. Mongolian, which stacks suffixes onto stems, has the same problem.

## Byte pair encoding for words {#bpe}
> Start from characters and repeatedly merge the most frequent adjacent pair of symbols into a new symbol.

BPE was a data compression algorithm. The paper turns it into a way to learn a vocabulary: count every adjacent pair of symbols in the training words, weighted by how often each word occurs, merge the most frequent pair everywhere, and repeat. Each merge adds one symbol to the vocabulary, so the number of merges sets the vocabulary size. Merges never cross word boundaries, and an end-of-word marker keeps "end of a word" distinct from "inside a word". This is the trainer you wrote in 1.5 and reuse in 4B.1.

## Open vocabulary {#open-vocab}
> Frequent words end up as single tokens, rare words as a few pieces, and any word at all can still be written with characters.

Because single characters stay in the vocabulary, nothing is ever unknown. The training corpus decides what gets merged: words and suffixes that appear often become one token, rare ones are spelled out in pieces the model has seen in other words. The model can then learn to translate a rare word from its parts.

## Choices that matter {#choices}
> The vocabulary size is a trade-off, and learning one joint vocabulary for both languages keeps shared names segmented the same way.

More merges mean shorter sequences but more rare tokens, each with fewer training examples. The paper compares learning BPE separately for the source and target language with learning it jointly on both. Joint BPE makes a name or loanword split the same way on both sides, which helps the model copy or transliterate it, including between Latin and Cyrillic script.

## What it achieves {#results}
> Subword models translate rare and unseen words better than word-level models with a back-off dictionary, and improve overall translation quality.

On WMT English→German and English→Russian, subword segmentation beat the word-level baseline, with most of the gain on rare words. Within a few years nearly every language model used some form of subword tokenizer, and byte-level BPE, as in Llama 3.1, is a direct descendant.

## How the course uses it {#course}
> 4B.1 trains a BPE on Mongolian text and measures how many tokens each word costs before and after.

Llama 3.1's BPE was learned mostly from English and other high-resource text, so its merges rarely fit Mongolian suffixes and a word costs about 3.4 tokens in 1.6's reference audit. Training a BPE on your 4A.3 corpus gives merges that do fit. Instead of replacing Llama's vocabulary, the lesson adds the Mongolian tokens that save the most, ranked by count × (pieces − 1).
