---
paper: Simple and Scalable Strategies to Continually Pre-train Large Language Models
gist: Re-warming and re-decaying the learning rate, plus replaying some of the old data, lets you continue pretraining on new data and match retraining from scratch on everything.
---

## The problem {#problem}
> When new data arrives, retraining from scratch is expensive, but simply continuing training adapts poorly or forgets what the model knew.

A pretrained model has finished its learning-rate schedule, usually ending at a small rate. Training on a new dataset from there learns slowly, and training hard on the new data alone degrades performance on the old distribution. The paper looks for a cheap recipe that avoids both.

## Re-warm, then re-decay {#schedule}
> Start the continued run with a fresh warmup to a high learning rate, then decay it again, as in a normal pretraining schedule.

A new warmup and cosine decay let the model adapt to the new data. The re-warming itself causes a temporary rise in loss on the old data, which is part of why the next ingredient matters. This is the shape of `lr_at` in 4B.3.

## Replay {#replay}
> Mixing a fraction of the original pretraining data into the new data controls forgetting.

The authors study a weak shift, between two English web corpora, and a strong one, from English to German. A small share of replayed old data is enough to protect the old distribution under the weak shift. The strong, language-changing shift needs a larger share. Adapting a model to Mongolian is a strong shift of the same kind.

## What it achieves {#results}
> With these two ingredients, continued pretraining matches a model retrained from scratch on the combined data, at a fraction of the compute.

They show this at 405M and 10B parameters. The paper also proposes schedules without a final decay, so a model can keep being updated as new data arrives without re-warming each time.

## How the course uses it {#course}
> 4B.3 re-warms and decays the learning rate, and its hints and the English check come from the replay result.

4B.3's run uses a short warmup and a cosine decay to a floor. It adds no English replay, because the control run added none and the two must match. Instead both runs track English perplexity, and the control's +5% is the reference: if you ever train longer on Mongolian, this paper says to mix English back in for both runs.
