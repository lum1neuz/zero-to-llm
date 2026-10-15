---
gist: A serving engine turns your model into a web service for many users at once. This lesson shows why one user's speed is set by memory bandwidth, why a batch of users rides almost free, and how to read a benchmark honestly.
minutes: 16
---

## What this lesson is for {#why}
> You've trained and quantized a Mongolian model. Now you put it behind a server and measure how fast it answers one user, and how many users it can serve at once.

Until now you've run your model from a Python script: one prompt in, one answer out. Real use looks different. Many people send prompts at the same time, each expects text to start appearing quickly, and the GPU costs money every hour whether it's busy or not.

Two open-source engines dominate this job, and they were built for different situations:

- **llama.cpp**, which you used in 3.6 for GGUF and k-quants. It runs anywhere: a laptop CPU, a Mac, a gaming GPU.
- **vLLM**, a Python server built for big GPUs that serve many users at once.

In this lesson you serve the same fp16 model with both and benchmark them with 1, 4 and 16 users at once. First, in the browser, you work out the arithmetic that predicts the result. The rest of phase 5 builds the pieces inside vLLM that make it fast under load: paged KV memory (5.2), speculative decoding (5.3) and continuous batching with prefix caching (5.4).

## What you need from before {#before}
> Generation is a loop: one forward pass per new token, reusing the cached keys and values of every earlier token.

A quick refresher:

- **Prefill and decode.** The first pass reads the whole prompt at once and fills the **KV cache** (lesson 2.2): the key and value vectors of every prompt token, in every layer. Every later pass, called **decode**, feeds in just the newest token. That token attends to everything in the cache and adds its own K and V.
- **Batching** (2.4): several sequences go through the model in one forward pass, stacked as rows of one tensor.
- **GGUF** (3.6): llama.cpp's file format. It holds the weights, the tokenizer and the chat template in one file, in fp16 or a k-quant like Q4_K_M.

One number to keep in mind: your model has about **8 billion parameters**, so in fp16 (2 bytes each) its weights take about 16 GB, or 15 GiB.

## What a serving engine does {#engine}
> A serving engine is an HTTP server with a model inside: it queues requests, batches them onto the GPU and streams tokens back as they're made.

If you've built a web backend, most of the shape is familiar:

| Web backend | LLM serving engine |
|---|---|
| HTTP endpoint | `/v1/chat/completions` (OpenAI-style) |
| Request queue | Prompts waiting for a slot on the GPU |
| Worker pool | Sequences in the current batch |
| Response body | Tokens, streamed one at a time |
| p50 / p99 latency | Time to first token, time per token |

The new part is the inner loop. Each **step** runs one forward pass over everyone currently being served and adds one token to each of their answers. The engine decides, step by step, who is in that batch. That decision is what separates a fast engine from a slow one.

## Two engines, two jobs {#two-engines}
> llama.cpp is built to run anywhere for one user; vLLM is built to keep a big GPU busy for many.

| | llama.cpp | vLLM |
|---|---|---|
| Written in | C++, one binary | Python + CUDA kernels |
| Runs on | CPU, Mac, NVIDIA, AMD | Mostly NVIDIA GPUs |
| Model format | GGUF, k-quants 2–8 bit | Hugging Face checkpoints |
| Best at | One user, small hardware | Many users, one big GPU |
| KV memory | Fixed slots per user | Paged blocks (5.2) |

With one user, both are limited by the same thing (next section), so they come out nearly tied. With 16 users, vLLM's scheduler and memory management pull ahead. In the lesson's sample run it makes about twice the tokens per second.

## Latency and throughput {#latency-throughput}
> Users feel latency: how long until text starts, and how fast it streams. Operators pay for throughput: tokens per second across everyone.

A serving benchmark records four things per request: when it **arrived**, when its **first** token came back, when it was **done**, and how many **tokens** it got. Three numbers come out of that:

- **TTFT, time to first token**: first − arrive. It covers waiting in the queue plus prefill.
- **TPOT, time per output token**: (done − first) / (tokens − 1). This is the streaming speed after text starts.
- **Throughput**: all tokens from all requests, divided by the wall-clock time of the run.

A worked example. A request arrives at 1.0 s, its first token comes at 1.3 s, it's done at 3.3 s and it got 101 tokens:

```
TTFT = 1.3 - 1.0 = 0.3 s
TPOT = (3.3 - 1.3) / 100
     = 0.020 s = 20 ms
```

Why 100 and not 101? The first token arrives at the *start* of the 2.0 s window. Only the 100 after it fall inside, like fence posts and gaps.

Latency and throughput pull against each other. A bigger batch means more total tokens per second, but each step takes a bit longer, so each user's TPOT rises a little.

## Why decode is memory-bound {#memory-bound}
> Every decode step reads all 16 GB of weights from GPU memory, and that read, not the math, sets the speed for one user.

An RTX 3090 has two limits, rounded from its spec sheet:

| Limit | RTX 3090 |
|---|---|
| Memory bandwidth | 936 GB/s |
| fp16 tensor-core math | 71 TFLOP/s |

For one user, a decode step has to:

1. **Read every weight once**: 8e9 parameters × 2 bytes = 16e9 bytes. At 936e9 bytes per second that's 0.0171 s, or **17.1 ms**.
2. **Do the math**: each weight does one multiply and one add per token, which is 2 FLOPs. 2 × 8e9 = 16e9 FLOPs at 71e12 per second is **0.23 ms**.

The GPU streams the next weights in while it multiplies the last ones, so the two overlap and the step takes as long as the *slower* one: 17.1 ms. That's at most 1000 / 17.1 ≈ **58.5 tokens per second**, and real engines get 48–50. The math units sit idle about 98% of the time.

Two classic mistakes the tests check for: forgetting the 2 bytes per parameter (giving 8.5 ms), and adding the two times instead of taking the larger one.

## Batching rides free, up to a knee {#roofline}
> A batch of users shares one read of the weights, so throughput grows with the batch until the math catches up.

Here's the key fact. If 8 users are in the batch, the GPU still loads each weight **once** and multiplies it against all 8 rows. The read stays at 17.1 ms. Only the math grows, 0.23 ms per user:

| Batch | Read | Math | Step | Tokens/s |
|---|---|---|---|---|
| 1 | 17.1 ms | 0.2 ms | 17.1 ms | 58 |
| 8 | 17.1 ms | 1.8 ms | 17.1 ms | 468 |
| 64 | 17.1 ms | 14.4 ms | 17.1 ms | 3,744 |
| 256 | 17.1 ms | 57.7 ms | 57.7 ms | 4,438 |

Until about batch 76, where the math time reaches 17.1 ms, every extra user is almost free: throughput grows in a straight line. After that the GPU is **compute-bound** and throughput flattens. This picture is called a **roofline**: a sloped line while memory is the limit, a flat roof once compute is.

Real engines reach the knee later than this, because attention also reads every user's KV cache and kernels never hit peak FLOPs. The shape holds, though, and it explains the whole phase: serving well means keeping the batch big.

## Reading a benchmark honestly {#benchmark}
> Report the median wait, time per token after the first, and tokens over the run's wall-clock span. Each has a tempting wrong version.

Benchmarks are easy to get subtly wrong. These are the mistakes the challenge tests catch:

| Number | Right | Tempting and wrong |
|---|---|---|
| TTFT | first − *its own* arrive | first − 0 (start of run) |
| p50 | median of the waits | mean, which one slow request pulls up |
| p50 | middle after sorting | middle of the list as recorded |
| TPOT | ÷ (tokens − 1) | ÷ tokens |
| TPOT | from first to done | from arrive to done |
| Throughput | ÷ (last done − first arrive) | ÷ sum of request durations |

The last row matters most. Requests overlap: while one streams, three others do too. Adding up their durations counts that time several times, and the throughput comes out a fraction of the truth.

Medians hide the unlucky. In the challenge's run one request waits 1.05 s for its first token while the others wait about 0.2 s. Dashboards also show p90 and p99 for exactly that reason.

## Same model, slightly different text {#greedy-match}
> Two engines can disagree on a few greedy outputs even with identical weights, because floating-point addition depends on order.

Greedy decoding always takes the top token, so you might expect both engines to produce identical text. Usually they do. Sometimes they don't, because their kernels add the same numbers in a different order, and float addition isn't associative:

```
# float32 has 24 bits of precision,
# so near 1e8 the spacing is 8
(1e8 + 1) - 1e8   →  0
(1e8 - 1e8) + 1   →  1
```

The differences are tiny, around the last digit. But when the top two tokens' logits are nearly tied, a tiny difference can flip which one wins, and every token after that follows a different path. The lesson's quality metric checks the first 64 greedy tokens on 50 prompts. 46 of 50 matching (92%) is healthy. Garbage text or mismatches from the very first token would mean something is really wrong, such as a different chat template.

## The GPU run {#gpu-run}
> You'll convert your model to an fp16 GGUF, serve it with each engine in turn, and benchmark 1, 4 and 16 concurrent users.

The steps, about an hour in all:

1. Convert your model to an fp16 GGUF with llama.cpp's `convert_hf_to_gguf.py`.
2. Start `llama-server` with `-np 16`: 16 parallel slots. `-c 16384` is split across them, so each user gets 1,024 tokens of context.
3. Benchmark it, stop it, then start `vllm serve` with fp16 and benchmark again. Don't run them side by side: vLLM claims about 92% of the GPU memory at startup.
4. Run the greedy check on 50 prompts.

What healthy output looks like, from the lesson's replay. The replay is **illustrative sample data written for the course, not a real recording**:

- vLLM's startup log reports about 15 GiB of weights and ~4.4 GiB of KV cache in **GPU blocks** of 16 tokens. That's lesson 5.2.
- At 1 user both engines make about 48–50 tok/s with similar TTFT.
- At 16 users llama.cpp's TTFT climbs into the hundreds of ms, while vLLM stays near 100 ms and makes about 2× the tokens.

You record two metrics: the greedy match percentage, and vLLM's throughput divided by llama.cpp's at 16 users.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Serving engine**: a server that runs a model for many clients, batching their requests on the GPU. Examples: vLLM, llama.cpp's `llama-server`.
- **Prefill**: the forward pass over the whole prompt that fills the KV cache.
- **Decode**: each later forward pass, which makes one new token per sequence.
- **Step**: one forward pass of the batch. In decode it adds one token to every sequence in the batch.
- **Batch**: the sequences processed together in one step.
- **TTFT (time to first token)**: from a request's arrival to its first token. Queueing plus prefill.
- **TPOT (time per output token)**: time between tokens after the first one.
- **Throughput**: total tokens per second across all users.
- **p50, p90, p99**: percentiles. p50 is the median: half the requests are faster.
- **Memory-bound**: limited by how fast data can be read from memory, not by arithmetic.
- **Compute-bound**: limited by arithmetic speed.
- **Bandwidth**: bytes per second the GPU can read from its memory (936 GB/s on a 3090).
- **FLOP**: one floating-point operation. A multiply-add is 2 FLOPs.
- **Roofline**: the model above. Speed grows with batch while memory-bound and flattens once compute-bound.
- **Concurrency, users**: how many requests are in flight at once.
- **GGUF**: llama.cpp's single-file model format.
- **Slot**: in llama.cpp, a fixed share of context reserved for one user (`-np` slots).
- **Greedy decoding**: always picking the most likely next token.
- **Near-tie**: two tokens whose logits are almost equal, so rounding can flip the choice.
- **Illustrative**: sample data written for the course, not a recording of a real run.
