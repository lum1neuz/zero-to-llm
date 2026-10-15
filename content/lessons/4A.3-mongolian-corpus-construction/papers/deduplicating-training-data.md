---
paper: Deduplicating Training Data Makes Language Models Better
gist: Common web training sets are full of exact and near-duplicate text. Removing it makes models memorise far less, makes evaluation more honest, and costs nothing in quality.
---

## The problem {#problem}
> Large web corpora repeat the same text many times, from boilerplate sentences to whole mirrored documents, and nobody had measured how much that matters.

Datasets like C4 are built by filtering a web crawl, and their filters weren't designed to catch repetition. The authors found single sentences repeated tens of thousands of times and many documents that differ only in a header or footer. Repeated text gets seen many times per epoch, so it gets more weight than anything else.

## Two kinds of duplicate {#two-kinds}
> Exact substring matching catches long repeated spans inside documents; approximate matching with MinHash catches whole documents that are nearly the same.

The exact method builds a suffix array over the whole dataset and finds every span over a length threshold that appears more than once. That catches repeated boilerplate even inside otherwise unique pages. The approximate method compares documents by their n-gram sets, using MinHash to estimate Jaccard similarity and keep the comparison cheap, and removes one of each pair above a threshold. 4A.3 builds a small version of the second method.

## Memorisation {#memorisation}
> Models trained on deduplicated data emit memorised training text about ten times less often.

The authors generated text without a prompt and checked how much of it copied training data verbatim. With the original datasets a noticeable share did. With deduplicated data it dropped by about an order of magnitude. Much of what a model "memorises" is the text it saw over and over.

## Train-test overlap {#overlap}
> Standard validation sets share a surprising amount of text with their training sets, which inflates the scores measured on them.

Because the same pages are mirrored, the validation split of a web dataset can contain near-copies of training documents. The paper found this in several standard datasets. A model can score well on that part of the validation set by remembering rather than generalising. That's why 4A.3 splits off the held-out set after dedup, not before.

## Quality and cost {#results}
> Deduplicated models reach the same or better perplexity, often in fewer training steps.

Removing duplicates didn't hurt language modelling quality and sometimes helped, while saving compute: there is simply less redundant text to train on. It's one of the rare changes that improves several things at once.

## How the course uses it {#course}
> 4A.3 applies the paper's near-duplicate idea to a Mongolian crawl, where mirrored news makes it matter even more.

Mongolian web text is scarce, so the same news story mirrored across many sites is a large fraction of it. Your pipeline does exact dedup first, then MinHash at Jaccard ≥ 0.7, and only then makes the held-out split that 4A.4 measures perplexity on.
