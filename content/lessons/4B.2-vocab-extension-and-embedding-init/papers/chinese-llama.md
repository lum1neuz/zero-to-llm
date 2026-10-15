---
paper: Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca
gist: Adding Chinese tokens to LLaMA's vocabulary, then continuing pretraining and instruction tuning with LoRA, gives a model that reads and writes Chinese far more efficiently.
---

## The problem {#problem}
> LLaMA's tokenizer had few Chinese tokens, so Chinese text fell apart into many short pieces, often single bytes.

LLaMA was trained mostly on English and other Latin-script languages. Its 32,000-token SentencePiece vocabulary contains only a small number of Chinese characters, and the rest are spelled out with byte tokens. That makes Chinese sequences long, slow to generate and hard to learn from, because the meaning of one character is spread over several tokens. It's the same situation as Llama 3.1 and Mongolian.

## Extending the vocabulary {#vocab}
> Train a Chinese SentencePiece model and merge its tokens into LLaMA's tokenizer, about 20,000 new tokens.

The authors trained a separate Chinese tokenizer on Chinese text and merged its pieces into the original vocabulary, keeping all of LLaMA's tokens and their ids. The extended tokenizer encodes the same Chinese text in roughly half as many tokens. 4B.1 does the same for Mongolian, but picks which tokens to add by how many tokens each one saves.

## Resizing the model {#resize}
> The embedding matrix and the LM head both grow by one row per new token; the old rows stay where they are.

Every new token needs a row in the input embedding and in the output layer. Old ids keep their rows, so the pretrained model still works on text that uses only old tokens. The new rows start untrained, and how you fill them decides where training starts, which is what 4B.2 measures with the step-0 loss.

## Continued pretraining {#pretraining}
> Train on Chinese text with LoRA on the transformer and the embeddings and LM head trained in full.

The new rows have to be learned from scratch, so the embeddings and the LM head are trainable, while the rest of the model is adapted through low-rank adapters to keep memory down. The paper describes a staged schedule, first adapting the embeddings with the transformer frozen and then training the adapters as well. 4B.3 follows the second form: new rows in full, everything else through the same LoRA as the control run.

## Instruction tuning {#instruction}
> The pretrained Chinese model is then instruction-tuned into an assistant, "Chinese Alpaca".

After continued pretraining, the model is fine-tuned on instruction and response pairs, again with LoRA, so it answers questions instead of only continuing text. That's the split between 4B.3 and 4B.4: the extended model alone, and the extended model plus instruction tuning.

## How the course uses it {#course}
> Phase 4B is this recipe, scaled down to one GPU and Mongolian, with a control run to say whether the new vocabulary is what helped.

The paper shows the recipe works for Chinese. It doesn't isolate the tokenizer as the cause. 4A.4's control run trains on the same text with the original tokenizer, so 4B.4's four-way comparison can tell how much of the gain comes from the vocabulary and how much from the extra Mongolian training.
