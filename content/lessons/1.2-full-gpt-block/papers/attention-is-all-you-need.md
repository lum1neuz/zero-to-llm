---
paper: Attention Is All You Need
gist: The Transformer layer is attention plus a per-position feed-forward network, each wrapped in a residual connection and layer normalisation, stacked six times on each side of an encoder–decoder.
---

## The layer {#layer ref="§3.1"}
> Every layer has two sublayers, multi-head attention and a feed-forward network, each with a residual connection around it.

An encoder layer is self-attention followed by a feed-forward network. A decoder layer adds a third sublayer that attends to the encoder's output. The base model stacks 6 layers on each side with width d_model = 512. A GPT block is the decoder layer without the encoder attention: masked self-attention, then the feed-forward network.

## Multi-head attention {#multi-head ref="§3.2.2"}
> Instead of one attention over d_model features, run h smaller attentions in parallel on separate projections and concatenate their outputs.

The base model uses h = 8 heads of d_k = 64 features each, so 8 × 64 = 512. Each head has its own query, key and value projections and can attend to different positions for different reasons. Because each head is narrower, the total cost is about the same as one full-width head. An output projection mixes the concatenated heads. Your `view(B, T, n_head, hs).transpose(1, 2)` is this split, and `c_proj` is the output projection.

## The feed-forward network {#ffn ref="§3.3"}
> Two linear layers with a ReLU between them, applied to every position separately and identically.

The inner layer is four times wider than the model, 2048 for d_model = 512. It doesn't mix positions; attention already did that. GPT-2 keeps the shape and swaps ReLU for GELU, which is the MLP you get in step 2.

## Residuals and layer normalisation {#residual-norm ref="§3.1"}
> Each sublayer's output is added to its input and the sum is normalised: LayerNorm(x + Sublayer(x)).

This is the post-norm layout: the residual stream itself is normalised after every sublayer. It works for 6 layers but gets harder to train as stacks grow deep. Later GPTs move the LayerNorm to the sublayer's input instead, which is the pre-norm block you build, and the `post_norm` mistake in step 2 is the paper's original layout.

## Position information {#positions ref="§3.5"}
> Attention ignores order, so the model adds a position signal to the token embeddings, here fixed sinusoids.

Without it, a shuffled sentence would produce the same set of outputs. The paper adds sines and cosines of different frequencies and reports that learned position embeddings work about as well. GPT-2 and the models in this course use learned embeddings, `wpe`.

## How the course uses it {#course}
> 1.2 builds the decoder-only version of this layer, pre-norm, and counts its parameters: 12·d² + 13·d.

Attention's four d × d projections (q, k, v and the output) and the feed-forward network's two d × 4d matrices make the 12·d². Biases and two LayerNorms add 13·d. Stacking 6 of these blocks between embeddings and an output layer gives the model you train in 1.3 and 1.4.
