---
paper: Fast Inference from Transformers via Speculative Decoding
gist: A small model guesses several tokens and the large model checks them all in one parallel pass, which speeds up decoding 2–3× while leaving the output distribution exactly unchanged.
---

## The problem {#problem}
> Autoregressive decoding makes one token per forward pass of the large model, and each pass is limited by memory bandwidth rather than compute.

Generating k tokens takes k serial passes of the big model. At small batch sizes each pass mostly streams weights from memory, so the GPU's arithmetic sits largely idle (5.1's roofline). The paper's observation: if you already had good guesses for the next few tokens, the big model could score all of them in a single pass for about the cost of one.

## Draft and verify {#draft-verify ref="§2"}
> A cheaper approximation model proposes γ tokens one at a time; the target model then evaluates all γ + 1 positions in parallel and keeps the agreeing prefix.

The approximation (draft) model runs autoregressively for γ steps. Then the target model runs once over the prefix plus the γ guesses, which gives its own distribution at every guessed position and one position beyond. The guesses are accepted left to right until the first rejection. The paper writes γ for what 5.3 calls k. The rejected position gets a token from the target's distribution, and if every guess is accepted, the extra position yields a bonus token. Every round therefore produces at least one token, so it's never slower than plain decoding in tokens per target pass.

## Speculative sampling {#sampling ref="§2.3"}
> Accept a drafted token x with probability min(1, p(x)/q(x)); on rejection, sample from the normalised max(0, p − q).

Here p is the target's distribution and q the draft's at that position. Tokens the draft over-proposes are thinned out, and the residual distribution adds back exactly the probability mass the draft under-proposed. The paper proves the resulting token is distributed exactly as p. So speculative decoding is lossless for any sampling setting, not just greedy, and needs no change to either model. You implement this rule in 5.3's `verify` challenge and check the output frequencies against p.

## Acceptance rate {#acceptance ref="§3.1"}
> The expected acceptance rate α is the overlap between the two distributions, Σ min(p, q), so a draft that's closer to the target is accepted more often.

α equals one minus the total variation distance between p and q. With guesses accepted independently at rate α, the expected tokens per target pass are 1 + α + … + α^γ = (1 − α^(γ+1)) / (1 − α). That's 5.3's `expected_tokens`.

## Choosing γ and the speedup {#speedup ref="§3.3"}
> With c the cost of a draft pass relative to a target pass, the expected walltime improvement is (1 − α^(γ+1)) / ((1 − α)(γc + 1)), which has a best γ for each α and c.

Each extra guess adds c to the round's cost but is less likely to survive than the one before it. So the speedup rises, peaks and falls as γ grows. A better draft (higher α) or a cheaper one (lower c) pushes the best γ up. The method also does more total arithmetic than plain decoding, since rejected guesses are wasted work. It pays off when compute is spare, as in memory-bound decoding at small batch sizes.

## Results {#results ref="§4"}
> On T5-XXL with a much smaller T5 as the draft, decoding ran about 2–3× faster with identical outputs.

The draft models in the experiments were far smaller than the target, so c was small, and acceptance rates were high on tasks like translation and summarisation. The paper also notes that very simple drafts, even n-gram models, give some speedup because c is almost zero.

## How the course uses it {#course}
> 5.3 builds greedy verification, the speculative sampling rule and the speedup arithmetic, then sweeps γ for a 1B draft and your 8B model.

The greedy challenge checks that the output equals plain greedy and counts target passes. The `verify` challenge implements the accept/reject rule with a seeded generator. The `speedup` challenge is the formula above. On the GPU, Mongolian text lowers α for an English-heavy 1B draft, which is why the best γ in the replay is small.
