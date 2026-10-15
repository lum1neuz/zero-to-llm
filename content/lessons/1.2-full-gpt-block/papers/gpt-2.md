---
paper: Language Models are Unsupervised Multitask Learners
gist: A large Transformer language model trained only to predict the next token on a broad web corpus can do tasks like reading comprehension, summarisation and translation zero-shot, without task-specific training.
---

## The idea {#idea ref="§1–2"}
> If a language model is trained on enough varied text, many tasks are just continuations of a well-chosen prompt.

Instead of fine-tuning a separate model per task, the paper asks how far plain next-token prediction can go. A question followed by "A:" or an article followed by "TL;DR:" turns a task into text to continue. The answer gets better steadily as the model gets bigger.

## WebText {#webtext ref="§2.1"}
> The training data is about 40 GB of text from web pages linked on Reddit with at least 3 karma, used as a rough quality filter.

Common Crawl was large but noisy. Using outbound Reddit links that people had upvoted gave a cleaner, more varied corpus of about 8 million documents. Wikipedia pages were removed to make test-set overlap easier to reason about.

## Byte-level BPE {#bpe ref="§2.2"}
> The tokenizer runs BPE on raw bytes, so any text can be encoded, with a vocabulary of 50,257 tokens.

Starting from bytes instead of Unicode characters keeps the base vocabulary at 256 and removes unknown tokens. The paper also stops merges from crossing character categories, so "dog." and "dog!" don't each become a separate token. This is the tokenizer family you build in 1.5 and audit in 1.6.

## The model {#model ref="§2.3"}
> Four sizes from 117M to 1.5B parameters, all GPT-style decoders with LayerNorm moved to the input of each sub-block.

The changes from GPT-1 are small but lasting: pre-norm sub-blocks, an extra LayerNorm after the last block, residual-layer weights initialised smaller (scaled by 1/√N for N residual layers), a 1024-token context and the larger vocabulary. The block you build in 1.2 follows this layout, with `ln_1` and `ln_2` in front of attention and the MLP.

## Results {#results ref="§3"}
> Zero-shot, the largest model set new results on most of the language-modelling benchmarks tested, and showed promising but far-from-competitive results on other tasks.

Language modelling improved the most. Reading comprehension, summarisation and translation worked, but behind systems trained for those tasks. The paper's point is the trend: every task improves with model size, and the largest model still underfits WebText.

## How the course uses it {#course}
> GPT-2's block is the block of every model in this course, from 1.2's Shakespeare model to 1.4's TinyStories run.

You build its pre-norm block, its MLP with GELU and its tied output layer, and count their parameters. 1.5 rebuilds its byte-level BPE in miniature.
