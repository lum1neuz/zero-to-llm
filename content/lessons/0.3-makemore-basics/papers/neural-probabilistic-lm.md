---
paper: A Neural Probabilistic Language Model
gist: Learn a vector for every word together with a neural network that predicts the next word from those vectors, so what the model learns about one word carries over to similar words.
---

## The problem {#problem}
> Counting n-grams runs into the curse of dimensionality: almost every long enough sequence you'll test on never appeared in training.

An n-gram model, like 0.3's bigram, stores one probability per context it has seen. Each extra word of context multiplies the number of possible contexts by the vocabulary size, so the counts get sparse fast. Smoothing and backing off to shorter contexts paper over the gaps, but the model still has no idea that "the cat is walking" says something about "a dog was running". Each context is its own row in the table, with nothing shared between rows.

## Words as vectors {#embeddings}
> Give every word a learned feature vector, so similar words end up close together and a sentence you've never seen can still look familiar.

Each word gets a vector of a few dozen to a hundred numbers: a distributed representation. The vectors start out random and are trained along with the rest of the model. Words that appear in similar contexts are pushed towards similar vectors, so the probability the model learns for one sequence spreads to sequences made of neighbouring words. That sharing is where it generalises better than counting.

## The architecture {#architecture}
> Look up the vectors of the previous n − 1 words, concatenate them, pass them through a tanh hidden layer, and end with a softmax over the whole vocabulary.

The embedding table C is shared by every position. The concatenated context vectors go into a hidden layer with a tanh, then an output layer produces one score per vocabulary word, and a softmax turns the scores into probabilities. The paper also tries direct connections from the embeddings to the output that skip the hidden layer. Every parameter, embeddings included, is trained by gradient-based optimisation of the log-likelihood of the training text.

## Training cost {#cost}
> The softmax over the full vocabulary dominates the compute, so training needed a lot of machines for its time.

Every prediction computes a score for every word in the vocabulary, tens of thousands of them, which costs far more than the hidden layer. The authors trained with data and parameter parallelism across many CPUs. Making that output layer cheaper became a research thread of its own.

## What it achieves {#results}
> On two text corpora, the neural model reaches clearly lower perplexity than the best smoothed n-gram models, and mixing the two helps further.

The comparison is with interpolated and back-off trigram models, the standard of the time. The neural model wins on its own, and averaging its predictions with a trigram model's does better still, which suggests the two make different mistakes.

## How the course uses it {#course}
> 0.3 builds the counting baseline the paper set out to beat; makemore part 2, the MLP, is this paper's model at character level.

Your bigram is the simplest n-gram. Your neural bigram already contains the key move: `one_hot(x) @ W` is just a row lookup, the same thing as an embedding table. Swap the 27 × 27 W for a 27 × 10 table C, concatenate three characters' vectors, add a tanh hidden layer, and you have the MLP that 0.4 initialises properly and that reaches about 2.1 on names. Transformers keep the embedding table and the softmax and replace the fixed window with attention.
