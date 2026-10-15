---
paper: LLM.int8()
gist: A handful of hidden dimensions grow huge activations in large transformers, and int8 matmuls stay lossless only if those dimensions are split off and computed in fp16.
---

## Why int8 breaks at scale {#at-scale}
> 8-bit inference that works on small models starts losing accuracy as models grow, and the cause sits in the activations, not the weights.

Quantizing weights alone is forgiving. A matmul in int8 also needs the activations in int8, and activations change with every input. The paper follows int8 quality as model size grows and finds that it holds up to a few billion parameters and then degrades. The rest of the paper explains that break and works around it.

## What an outlier feature is {#outlier-features}
> An outlier feature is a hidden dimension with magnitudes of 6 or more that shows up in most layers and a noticeable share of tokens.

The authors call a dimension an outlier feature when its magnitude reaches at least 6, it shows up in at least a quarter of the layers, and it appears in at least 6% of the sequence positions. Isolated spikes don't count. In small models such features are scattered and inconsistent. Around 6.7B parameters they become systematic: the same few dimensions are large across nearly all layers and most tokens, with values far above everything else.

## They matter to the model {#they-matter}
> The outlier dimensions are few, but removing them badly damages the model, so clipping is not an option.

Zeroing these few dimensions makes perplexity much worse, even though they're a tiny fraction of the features. That rules out the cheap fix of clipping activations to a fixed range. Whatever quantization scheme you use has to carry those values through accurately.

## Why per-token scales aren't enough {#vector-wise-limit}
> Vector-wise quantization gives each token its own scale, but an outlier present in every token still sets every token's scale.

Vector-wise int8 scales each row of X (a token) and each column of W (an output) separately, then multiplies the int32 result by both scales. That contains a weight outlier to one column. An activation outlier is different: it sits in the same feature for almost every token, so it inflates almost every row's scale, and that row's ordinary features land on a few levels. This is the failure you reproduce with `int8_matmul` in 3.3.

## Mixed-precision decomposition {#decomposition}
> Split the inner dimension: outlier feature columns multiply in fp16, all other columns multiply in int8, and the two partial products are added.

A matmul sums over features, so you can compute it as two sums over disjoint feature sets. Pick out the columns of X where any value passes the threshold, take the matching rows of W, and multiply them in fp16. Quantize everything else vector-wise and multiply it in int8. Add the results. Only a tiny share of the values go through fp16, so almost all the arithmetic stays int8 and the outliers keep their precision. That is `mixed_matmul` in 3.3.

## What it costs and buys {#tradeoff}
> The split keeps fp16 quality for very large models at about half the weight memory, but the extra work makes it no faster than fp16.

With the decomposition, models up to 175B parameters keep their fp16 perplexity and zero-shot accuracy with int8 weights. The price is speed. Finding outlier columns, gathering them and running two matmuls adds overhead, which is noticeable on smaller models. LLM.int8() is a memory saving, and later methods like SmoothQuant aim to keep the quality while staying in pure int8 kernels.

## Where it sits in the course {#course}
> 3.1 used this paper's absmax core; 3.3 is about its central finding, the outliers, and its fix, the decomposition.

3.1 quantized weights per row with absmax and never touched activations. 3.3 adds activations, shows an outlier feature wrecking per-token scales on a 4 × 6 example, and builds the decomposition. SmoothQuant, the lesson's other paper, attacks the same outliers by moving their scale into the weights instead of splitting the matmul.
