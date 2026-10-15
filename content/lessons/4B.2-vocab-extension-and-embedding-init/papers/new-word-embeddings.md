---
paper: Initializing New Word Embeddings for Pretrained Language Models
gist: Start new embedding rows at the average of the existing ones, not at random, and the model's predictions barely change when the vocabulary grows.
---

## The problem {#problem}
> Randomly initialised rows for new words can make a pretrained model's predictions much worse before it has trained on a single example.

When you add words to a pretrained language model, the new rows of the output layer each produce a logit at every position. If those rows are random, some of them produce large logits by chance, softmax gives those new words probability they haven't earned, and that probability is taken from the words that should be predicted. The loss goes up on all text, even text that never uses a new word.

## Measuring the damage {#kl}
> Compare the model's next-word distribution before and after adding words, with the KL divergence between the two.

The note frames initialisation as a question you can answer before training: how far does the distribution over the old vocabulary move when the new rows are added? A good initialisation keeps that distance small, so fine-tuning starts from the model you had rather than from a damaged one.

## Average the existing embeddings {#average}
> If every new row equals the mean of the old rows, each new word gets roughly an average logit and can't dominate the softmax.

With the new rows at the mean, a new word's logit is the average of the old words' logits at that position, which is never larger than the largest old logit. The note shows this bounds how much probability the new words can take, so the change to the model's distribution stays small. Adding a little noise, drawn with the mean and covariance of the old embeddings, keeps the new rows distinct while staying close to that average.

## Where it shows up {#adoption}
> Hugging Face transformers later made this the default when you resize token embeddings.

Recent versions of `resize_token_embeddings` initialise new rows from a normal distribution with the old embeddings' mean and covariance instead of a small random init. That's the "mean of all rows" bar in 4B.2's replay.

## How the course uses it {#course}
> 4B.2 checks the claim at step 0 and goes one step further, starting each new token from the mean of its own pieces.

In the toy model and the Llama replay, random rows start worse than a uniform guess, and the mean of all rows starts much better. Each Mongolian token you add was already being spelled with known Llama pieces, so averaging those particular rows gives it a start that reflects what the word is, and the step-0 loss is lower still.
