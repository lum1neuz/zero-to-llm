---
paper: The Curious Case of Neural Text Degeneration
gist: Picking the most likely text makes language models dull and repetitive, and sampling from the full distribution makes them incoherent, so sample only from the smallest set of tokens that covers most of the probability (nucleus, or top-p, sampling).
---

## The puzzle {#puzzle}
> A model that writes fluent text when you sample from it produces bland, looping text when you ask for its most likely output.

Greedy decoding and beam search look for high-probability text, which sounds like the right goal. In practice, open-ended generation with them quickly falls into repetition, the same "the the the" loop your tiny model shows in 2.1. The authors use GPT-2 to study why, and what to do instead.

## Human text isn't the most likely text {#likelihood}
> Real human writing has lower per-token probability than beam-search output, and its probability jumps around from token to token.

When you score human text under the model, it isn't concentrated at the highest-probability choices. People regularly use words the model finds somewhat surprising. Maximizing probability therefore moves generations away from what human text looks like. Repetition also feeds on itself: once a phrase has been repeated, the model rates repeating it again as even more likely.

## The unreliable tail {#tail}
> Pure sampling avoids repetition but keeps picking tokens from the long tail of unlikely words, and one bad pick can derail the rest.

The vocabulary has tens of thousands of tokens. Each tail token is unlikely, but together they can hold a meaningful share of the probability, so plain sampling lands there often. The model's estimates for those rare tokens are also the least reliable. That's the T = 1.5 output in 2.1 and the noisy T = 1.0 line in 2.3.

## Why top-k isn't enough {#top-k}
> A fixed k is too small when the model is unsure and too large when it's confident.

Top-k sampling keeps the k most likely tokens. When the distribution is flat, many reasonable tokens fall outside the top k. When it's peaked, the top k still includes unlikely tokens that share a little probability. The right number of candidates changes from step to step, and a constant k can't follow it.

## Nucleus sampling {#nucleus}
> Keep the smallest set of top tokens whose probabilities add up to p, renormalize, and sample from that set.

At each step, sort the tokens by probability and add them up from the top until the total reaches p (for example 0.9). Only those tokens can be drawn. When the model is confident, the nucleus may be one or two tokens. When it's uncertain, it widens. This is the `top_p` you write in 2.3: the token that takes the running total past p stays in, and everything after it is cut to `-inf`.

## How they compare decoders {#evaluation}
> They judge generations by how closely their statistics match human text, not by likelihood alone, and nucleus sampling comes closest.

The paper compares decoding methods on several properties: perplexity of the generated text, how much it repeats itself, how varied different samples are, and how word frequencies are distributed. Text that's too likely is repetitive and samey, and text that's too unlikely is incoherent. Nucleus sampling sits closest to human text across these measures.

## How the course uses it {#course}
> 2.3 builds top-k and top-p and measures repetition with distinct-2, a simple version of the paper's diversity checks.

distinct-2 counts unique neighbouring token pairs. It catches greedy repetition well, but like the paper's point about likelihood, one number isn't enough: very high distinct-2 often means noise. That's why the metric step asks you to read samples too.
