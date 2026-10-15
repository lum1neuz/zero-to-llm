---
paper: Attention Is All You Need
gist: A sequence model built only from attention, with no recurrence or convolution, translates better than the recurrent models before it and trains much faster because every position is processed in parallel.
---

## The problem with recurrence {#problem ref="§1"}
> RNNs read a sequence one step at a time, so training can't be parallelised along the sequence and distant words are many steps apart.

Before this paper, the best translation models were recurrent encoder–decoders with attention bolted on. A recurrent layer has to finish position t − 1 before it can start position t, which leaves a GPU waiting on long sequences. Information from far back also has to survive many updates to reach the current position. The authors keep the attention and drop the recurrence.

## Scaled dot-product attention {#scaled ref="§3.2.1"}
> Each output is a weighted average of value vectors, with weights from softmax(q·k / √d_k).

Queries, keys and values are vectors. Every query is dotted with every key, the scores are divided by √d_k, softmax turns each row into weights, and the weights mix the values. All of it is two matrix multiplies and a softmax, which is why it's fast on a GPU. This is the `causal_weights` function you write in step 1, plus the final `@ v`.

## Why divide by √d_k {#why-scale ref="§3.2.1"}
> Dot products of d_k-dimensional vectors grow with √d_k, and large scores push softmax into regions with tiny gradients.

The paper's argument: if the entries of q and k have mean 0 and variance 1, their dot product has variance d_k. Without the scale, scores for large d_k are big, softmax becomes nearly one-hot, and its gradients become very small. The Learn card that asks how big q·k is for d = 64 is this argument with numbers.

## Masking the future {#masking ref="§3.2.3"}
> In the decoder, scores for positions after the current one are set to −∞ before the softmax, so each position only sees what came before it.

The decoder is trained on whole target sentences at once, with every position predicting its next token. Without a mask a position could simply read its own answer. Setting the illegal scores to −∞ makes their softmax weight exactly 0. GPT keeps only this masked, decoder-style attention, which is the version you build in this lesson.

## Self-attention against recurrence {#comparison ref="§4"}
> Self-attention connects any two positions in one step, at a cost that grows with the square of the sequence length.

The paper compares layer types on three counts: work per layer, how much can run in parallel, and the longest path between two positions. Self-attention wins on the last two. Its weak spot is the n × n score matrix, which is fine for sentences but expensive for long contexts. That cost comes back in phase 2 (the KV cache) and in fused attention kernels.

## How the course uses it {#course}
> 1.1 builds one masked attention head by hand; 1.2 adds the paper's multiple heads, feed-forward layer and residuals.

You write scaled dot-product attention and a causal mask on tiny tensors, then wrap it in a head with learned key, query and value projections. The encoder, the cross-attention between encoder and decoder, and the positional encodings aren't needed here: a GPT is a stack of masked self-attention blocks with learned position embeddings.
