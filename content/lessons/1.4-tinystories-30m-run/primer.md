---
gist: Before a long GPU run, count the model's parameters and estimate its compute bill on paper, then train it and judge the result by its perplexity.
minutes: 16
---

## What this lesson is for {#why}
> You train the 30M-parameter GPT that every quantization lesson in phase 3 works on, and learn to size a run before you pay for it.

So far your models trained in seconds or minutes. This one takes about 45 minutes on a good GPU and a few hours on a free Colab T4. That's long enough that you want to know, before you press Enter, how big the model is, how much compute it needs and roughly how long it will take.

So the lesson goes in that order. First you count the model's parameters from its config alone. Then you turn that count into FLOPs and GPU-hours. Then you train it on **TinyStories**, a dataset of simple children's stories that small models can learn well, and record how good the result is.

If you fall behind schedule, this is the third lesson in the cut order. Lessons 3.1 and 3.2 then quantize the course's reference checkpoint instead of yours, so phase 3 still works.

## What you need from before {#before}
> 1.2's block size formula, 1.3's schedule and clipping, and the loss you've been watching since 0.3.

- **One block** of width d holds 12·d² + 13·d parameters (you derived it in 1.2).
- **Warmup, cosine decay and clipping** from 1.3 are all switched on in this run.
- **Loss** is the average negative log likelihood per token, in nats. Lower is better.

## The config {#config}
> Six blocks, 512 wide, 8 heads, 512 tokens of context and a 21,504-token vocabulary.

| Field | Value | Meaning |
|---|---|---|
| `n_layer` | 6 | blocks stacked |
| `n_head` | 8 | heads per block, 64 features each |
| `n_embd` | 512 | features per token, d |
| `block_size` | 512 | longest context, T |
| `vocab_size` | 21,504 | different tokens |

The vocabulary comes from a BPE tokenizer trained on TinyStories, the algorithm you write in 1.5. 21,504 is a multiple of 64, which keeps the matrix sizes friendly for the GPU.

## Where the parameters live {#params}
> Two embedding tables, six blocks and a final LayerNorm. The output layer reuses the token table, so it costs nothing extra.

- **wte**, the token embedding: one d-vector per vocabulary entry, a `[21504, 512]` table, about 11.0M numbers.
- **wpe**, the position embedding: one d-vector per position, `[512, 512]`, about 0.26M.
- **6 blocks**: about 18.9M together.
- **ln_f**, a final LayerNorm before the output, with a gain and a bias.

The output layer, `lm_head`, turns each token's final vector into one score per vocabulary entry. That needs a `[21504, 512]` matrix too. GPT-2 doesn't learn a second one: it reuses `wte`'s matrix. This is **weight tying**. It's sensible because both matrices answer the same question, "which vector goes with which token". Untying would add another 11.0M parameters, making the model about 36% bigger without adding a single block.

Your first challenge counts each group from the config. The total is 30,187,520, the "30.2M params, 60.4 MB in fp16" that 3.1's recording loads. The tests check each group by changing one setting and watching your count: one more vocab entry should add exactly 512, one more position 512, and one more block a full block's worth. Changing `n_head` should add nothing, because heads only cut the attention's features into slices.

## FLOPs: counting the work {#flops}
> Training costs about 6 floating-point operations per parameter per token: 2 in the forward pass and about 4 in the backward pass.

A **FLOP** is one floating-point operation, an add or a multiply. In the forward pass each weight is used once per token, in one multiply and one add, so that's 2 FLOPs per parameter per token. The backward pass computes two sets of gradients, for the activations and for the weights, and costs about twice the forward. That gives the rule of thumb:

```
training FLOPs ≈ 6 × N × D
  N = parameters
  D = training tokens
```

A small example: a 1M-parameter model trained on 1B tokens costs 6 × 10⁶ × 10⁹ = 6 × 10¹⁵ FLOPs. The rule leaves out attention's own score computations and a few other small costs, so it's slightly low, but it's the standard first estimate.

## From FLOPs to hours {#hours}
> Divide by the FLOPs per second the GPU really sustains, which is its peak times a fraction called MFU.

A GPU's spec sheet gives its **peak** speed in TFLOPS, trillions (10¹²) of FLOPs per second. An RTX 3090 is rated at about 71 TFLOPS for bf16 matrix math. Real training never reaches the peak. The fraction it does reach is called **MFU**, model FLOPs utilisation. A model this small manages about 25–30%: its matrices are small enough that time goes to memory traffic, small kernels and the data loader rather than the tensor cores.

For this run:

```
tokens = 15,000 steps
       × 32 sequences
       × 512 tokens
       = 245.8M

FLOPs  = 6 × 30.19M × 245.8M
       ≈ 4.45 × 10¹⁶

speed  = 71 × 10¹² × 0.30
       ≈ 2.13 × 10¹³ per second

time   ≈ 2,090 s ≈ 0.58 h
```

The recording took 0.71 h: evals, data loading and the FLOPs the rule leaves out. Treat the estimate as a floor. The two classic slips are forgetting that TFLOPS means 10¹² (your answer is off by a trillion) and multiplying by MFU instead of dividing (MFU makes the run *slower*).

## The data {#data}
> TinyStories is tokenized once into a file of uint16 ids, about 945 MB, and the run reads about half of it.

`prepare.py` tokenizes all 2.1 million stories once and writes the token ids to disk. With 21,504 tokens, every id fits in a **uint16**, an unsigned 16-bit integer that holds 0 to 65,535 in 2 bytes. 472.6M training tokens × 2 bytes ≈ 945 MB. Stored as int64, PyTorch's default for indices, it would be four times bigger for no benefit. The training script reads the file with `np.memmap`, so it never has to fit in RAM at once.

Each step uses 32 sequences of 512 tokens, 16,384 tokens. 15,000 steps make 245.8M tokens, about 0.52 of the dataset. An **epoch** is one full pass over the data, so this run sees most stories once and none twice.

## Perplexity {#ppl}
> Perplexity is e raised to the loss. It reads as "how many tokens the model is choosing between, on average".

Loss is measured in nats, which is hard to picture. Perplexity turns it into a count:

| Loss | Perplexity | Reading |
|---|---|---|
| 9.98 | 21,504 | uniform guess over the vocab |
| 3.52 | 33.8 | still very unsure |
| 2.30 | 10.0 | the pass limit |
| 2.06 | 7.84 | the recording's final value |

At step 0 the loss should be close to ln(21,504) ≈ 9.98: a random model spreads its bets evenly over every token. A final perplexity of 7.84 means the model is as unsure as if it were choosing uniformly among about 8 tokens at each step, out of 21,504.

## Health checks before a long run {#checks}
> Check the step-0 loss, overfit one batch, and watch the gap between train and val loss.

- **Step-0 loss near ln(vocab_size).** Much higher means the output layer starts out overconfident; scale its initial weights down.
- **Overfit one batch.** 200 steps on a single batch should drive the loss well under 1. If it can't, the bug is in the model or the training loop, not the data.
- **Watch train and val together.** A small gap is healthy. Val rising while train keeps falling means memorisation. The recording ends with train 1.98 and val 2.06, a small gap, which is what half an epoch of fresh text looks like.
- **Watch the grad norm** from 1.3. A steady climb means the learning rate is too high.

## What the run looks like {#run}
> Perplexity falls from about 34 at step 1,000 to 7.84 at step 15,000, crossing the pass limit of 10 a little before step 5,000.

The recording is **illustrative**: sample data written for the course, consistent with the 30M baseline in 3.1 and 3.2, not a real recording. It shows the whole flow:

1. `prepare.py` tokenizes TinyStories (about 945 MB on disk).
2. `train_gpt.py` prints the config, the parameter count and the budget estimate, then the step-0 loss of 10.02.
3. Every 1,000 steps it prints train loss, val loss, perplexity and throughput, about 96,400 tokens per second.
4. It saves `runs/tinystories-30m/best.pt`, the checkpoint phase 3 loads, and reports 0.71 GPU-hours.
5. `sample.py` writes a short story from "Once upon a time".

On a Colab T4, expect 3–4 hours. The T4 has no bf16 tensor cores, so use fp16 with a GradScaler, or fp32 with a smaller batch.

## Your metrics {#metrics}
> Val perplexity (pass under 10) and the GPU-hours the run took (pass under 5).

Perplexity is printed at the end of the run. If you only have the loss, compute e^loss. GPU-hours are the run's wall-clock time on one GPU, in hours. The reference values, 7.84 and 0.71, come from the illustrative recording.

## Words you'll see {#glossary}
> The terms in this lesson's cards, challenges and metrics.

- **Config**: the settings that define the model's size: n_layer, n_head, n_embd, block_size, vocab_size.
- **wte, wpe**: the token embedding and position embedding tables.
- **lm_head**: the output layer that scores every vocabulary entry.
- **Weight tying**: lm_head reuses wte's matrix instead of learning its own.
- **ln_f**: the final LayerNorm before lm_head.
- **FLOP**: one floating-point add or multiply. TFLOPS is 10¹² per second.
- **6·N·D**: training compute ≈ 6 × parameters × tokens.
- **Peak TFLOPS**: the GPU's rated maximum speed.
- **MFU**: the fraction of peak speed training really achieves.
- **GPU-hour**: one GPU busy for one hour.
- **Token**: one unit of text the model reads, a piece of a word or a whole word.
- **uint16**: an unsigned 16-bit integer, 0 to 65,535, 2 bytes.
- **Epoch**: one full pass over the training data.
- **Perplexity (ppl)**: e^loss, roughly how many tokens the model is choosing between.
- **Checkpoint**: the saved weights of a model, here `best.pt`.
- **Cut order**: which lessons to drop first if you fall behind; this one is third.
