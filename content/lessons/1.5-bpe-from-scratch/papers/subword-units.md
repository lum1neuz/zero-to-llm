---
paper: Neural Machine Translation of Rare Words with Subword Units
gist: Splitting words into subword units learned with byte-pair encoding lets a translation model with a fixed vocabulary handle rare and unseen words, and it translates them better than a word-level model with a back-off dictionary.
---

## The rare-word problem {#problem ref="§1"}
> Translation models had fixed vocabularies of common words, and everything else became an unknown token.

A vocabulary of tens of thousands of words still misses names, compounds and rare inflections, and those are often the words that matter. Earlier systems replaced unknown words with a placeholder and patched the output with a dictionary afterwards. The authors want the model itself to translate any word.

## Words are made of smaller units {#intuition ref="§2"}
> Many rare words can be translated piece by piece: names by their characters or syllables, compounds by their parts, cognates by regular sound and spelling changes.

A German compound can be translated part by part, and a name can often be copied or transliterated letter by letter. If the model's units are smaller than words, it can learn these regularities and apply them to words it has never seen.

## Byte-pair encoding {#bpe ref="§3.2"}
> Start from characters and repeatedly merge the most frequent adjacent pair into a new symbol, until the vocabulary reaches the size you want.

BPE was originally a data-compression method. Here, each word starts as a sequence of characters with an end-of-word marker. Count all adjacent symbol pairs across the training text, merge the most frequent, and repeat. Common words end up as single symbols and rare words as a few frequent pieces. The number of merges is the one setting that controls vocabulary size. Encoding new text applies the learned merges in the order they were learned, which is what your `encode` does.

## Joint vocabularies {#joint ref="§3.2"}
> Learning one set of merges on source and target text together keeps the segmentation of names and cognates consistent across the two languages.

If English and German learned separate merges, the same name might be split differently on each side, which makes copying it harder. A joint BPE gives both languages the same pieces wherever their spellings agree.

## Results {#results ref="§4"}
> On WMT 2015 English→German and English→Russian, subword models beat a word-level baseline with a back-off dictionary, especially on rare words.

Most of the gain is on rare and unseen words, where the subword models produce correct translations the baseline could only patch. The models are also simpler, with no separate dictionary step.

## How the course uses it {#course}
> 1.5 builds the same merge loop, but on bytes, as GPT-2 does, so nothing is ever unknown.

The paper starts from characters within words. Byte-level BPE starts from the 256 byte values and needs no end-of-word marker, so any text in any script can be encoded. That's why your tokenizer round-trips Mongolian it never trained on, and why 1.6 finds those Mongolian words split into many small pieces.
