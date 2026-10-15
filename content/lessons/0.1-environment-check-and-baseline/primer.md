---
gist: Before you train anything, check two numbers. How fast your GPU multiplies matrices tells you it's really doing the work, and the loss of a model that knows nothing, ln 27 ≈ 3.30, is the bar every later model has to beat.
minutes: 14
---

## What this lesson is for {#why}
> Two checks before any training: your GPU really does the arithmetic, and you know what score "knowing nothing" gets.

In 0.0 you installed Python and PyTorch and ran `torch.cuda.is_available()`. This lesson goes one step further.

1. **Is the GPU doing the work, and how fast?** You'll time one big matrix multiplication on the CPU and on the GPU, in a few number formats. Every model in this course spends most of its time on exactly that operation, so this one number predicts how long your training runs will take.
2. **What does a useless model score?** Makemore, which you build in 0.3, predicts the next letter of a name. Before it learns anything, a fair guess gives every one of the 27 symbols the same chance. You'll compute the score of that guess, called the **baseline**. Later models are judged by how far below it they get.

The browser challenge is the second check, with a few names. The first check needs your GPU, so you run a short script on your own machine, and the lesson shows an illustrative example of its output.

## What you need from before {#before}
> Tensors, shapes and dtypes from 0.0, plus one habit: averaging over a dimension with `.mean()`.

- A **tensor** is a grid of numbers with one **shape** (like `[3, 4]`) and one **dtype** (like `float32`).
- Python floats become `float32` tensors. Integer tensors can't be averaged until you call `.float()`.
- Methods like `.sum()` and `.mean()` with no `dim` use every number in the tensor and return a single number, a tensor of shape `[]`. `.item()` turns that into a plain Python float.

```python
p = torch.tensor([0.5, 0.25])
p.mean()         # tensor(0.3750)
p.mean().item()  # 0.375
```

## Why a GPU {#gpu}
> A CPU has a few fast cores; a GPU has thousands of simple ones. Matrix multiplication splits into thousands of independent pieces, so the GPU wins by a wide margin.

A **matrix multiplication** (matmul), written `a @ b` in PyTorch, combines two grids. Each cell of the result is a **dot product**: take a row of `a` and a column of `b`, multiply them position by position, and add up the products.

```
a = [[1, 2],    b = [[5, 6],
     [3, 4]]         [7, 8]]

(a @ b)[0, 0] = 1·5 + 2·7 = 19
(a @ b)[0, 1] = 1·6 + 2·8 = 22
(a @ b)[1, 0] = 3·5 + 4·7 = 43
(a @ b)[1, 1] = 3·6 + 4·8 = 50
```

No cell needs any other cell's answer, so all of them can be computed at the same time. That's the kind of work a GPU is built for. A rough analogy from the web: a CPU is one very fast server thread; a GPU is a farm of thousands of slow workers that all get one small job each.

Neural networks are mostly matmuls, millions of them. Lesson 0.3's model is one matmul, and a transformer is dozens per layer.

## Counting the work: FLOPs {#flops}
> An n×n matmul does about 2·n³ floating-point operations, so a 4096×4096 one is about 1.37 × 10¹¹ of them.

A **FLOP** is one floating-point operation: one multiply or one add. Count them for an n×n matmul:

- Each output cell is a dot product of length n: n multiplies and about n adds, so about **2n** operations.
- There are n × n = **n²** output cells.
- In total: 2n × n² = **2n³** operations.

For the 2×2 example above that's 2 · 2³ = 16 (8 multiplies, plus 4 adds that the formula rounds up to 8). For the benchmark size n = 4096:

```
2 · 4096³ = 137,438,953,472
          ≈ 1.374 × 10¹¹ FLOPs
```

Speed is **FLOP/s**, operations per second. The numbers are huge, so they're quoted in **TFLOP/s**: tera, 10¹², a million million per second.

```
TFLOP/s = FLOPs / seconds / 10¹²
```

Worked example from the lesson's cards: one fp16 matmul takes 2.01 ms = 0.00201 s.

```
1.374 × 10¹¹ / 0.00201
  ≈ 6.84 × 10¹³ FLOP/s
  = 68.4 TFLOP/s
```

The step that trips people up is the milliseconds: divide by 0.00201, not by 2.01.

## Number formats on the GPU {#dtypes}
> fp32 uses 4 bytes per number; fp16 and bf16 use 2 and run on the GPU's tensor cores, which makes them several times faster.

You met `float32` (fp32) in 0.0. GPUs also do arithmetic in two 16-bit formats:

| Format | Bytes | Roughly | Used for |
|---|---|---|---|
| `float32` (fp32) | 4 | 7 significant digits, huge range | the default; safe everywhere |
| `float16` (fp16) | 2 | 3 digits, small range (max 65,504) | fast inference; training needs care |
| `bfloat16` (bf16) | 2 | 2–3 digits, fp32's range | fast training on recent GPUs |

Recent NVIDIA GPUs have **tensor cores**, circuits built for small matmuls in 16-bit formats. On those, fp16 and bf16 run around three times faster than fp32. The lesson's illustrative numbers for an RTX 3090:

| Where | Format | Time per 4096² matmul | TFLOP/s |
|---|---|---|---|
| CPU | fp32 | 182.40 ms | 0.8 |
| GPU | fp32 | 6.21 ms | 22.1 |
| GPU | fp16 | 2.01 ms | 68.4 |
| GPU | bf16 | 2.03 ms | 67.7 |

Your card's numbers will differ. What matters is the pattern: the GPU is tens of times faster than the CPU, and 16-bit is a few times faster than fp32. Older cards, such as the T4 on free Colab, have no bf16 tensor cores; use fp16 there.

## Timing GPU code honestly {#timing}
> GPU calls return before the work is done, so you must wait for the GPU before reading the clock, and skip the slow first run.

When Python runs `b = a @ a` on a GPU tensor, it puts the job in a queue and moves on at once. It's like calling `fetch()` in JavaScript: you get control back immediately, and the answer arrives later. Timing that line alone measures how long it took to *queue* the work, which is why the cards show a "result" of 40 µs, an impossible 3,400 TFLOP/s.

Two rules fix it:

1. **Wait for the GPU.** `torch.cuda.synchronize()` blocks until every queued job has finished, like `await`. Call it before starting the clock and again before stopping it.
2. **Warm up first.** The first CUDA call in a program loads code onto the GPU and picks algorithms, which is slow. Run the matmul a few times untimed, then time the average of many runs.

## The environment check {#env-check}
> You run one short script on your machine: it prints your PyTorch and CUDA versions, your GPU's name, a timing table, and the baseline loss over all of names.txt.

The script is in the lesson's "Check your GPU" step, under "Run it yourself". It needs `names.txt`, a list of 32,033 real names (download link in the script). It does three things:

1. Prints `torch.__version__`, `torch.version.cuda` and the GPU's name, and stops with a clear message if no GPU is visible.
2. Times a 4096×4096 matmul on the CPU once, then on the GPU in fp32, fp16 and bf16, and prints milliseconds and TFLOP/s for each.
3. Computes the uniform baseline over every name in the file.

The lesson plays an **illustrative** example of this output. It was written for the course to show what healthy looks like; it is not a recording of a real run.

Healthy output: a GPU name, every GPU row in single-digit milliseconds, fp16/bf16 a few times faster than fp32, and a final baseline of 3.2958. Unhealthy output and what it means:

| You see | Likely cause |
|---|---|
| "No GPU" error | CPU-only PyTorch: `torch.version.cuda` prints `None`. Reinstall with the CUDA command from pytorch.org. |
| GPU rows close to the CPU row | The work isn't on the GPU, or the timing skips `synchronize()`. |
| Numbers that jump between runs | A laptop GPU heating up and slowing down. Run more repetitions. |

You record your best fp16 or bf16 TFLOP/s as the lesson's systems metric. Anything above 5 passes: a CPU manages about 1.

## What a language model predicts {#predict}
> At each step the model gives a probability to every possible next character. The one number we care about is the probability it gave to the character that actually came next.

A **probability** is a number from 0 to 1 that says how likely something is. The probabilities of all possible outcomes add up to 1. Makemore has 27 possible next characters (a–z and "."), so at every step it produces 27 probabilities that sum to 1.

Take the name `emma`. Wrapped in "." markers it is `.emma.`, which asks the model for five predictions:

```
after "."  → should say "e"
after "e"  → should say "m"
after "m"  → should say "m"
after "m"  → should say "a"
after "a"  → should say "."
```

So a name of length L gives L + 1 predictions. For each prediction you look up the probability the model gave to the right answer. A good model gives the right answers high probabilities. The loss turns all of those into one score.

## Logarithms, gently {#logs}
> The natural log, ln, turns "how likely" into "how surprised": ln 1 = 0, and the smaller the probability, the more negative its log.

The **natural logarithm** `ln(x)` answers "e to what power gives x?", where e ≈ 2.718. You won't need that definition. You need how it behaves on probabilities:

| p | ln p | −ln p |
|---|---|---|
| 1 | 0 | 0 |
| 0.9 | −0.105 | 0.105 |
| 0.5 | −0.693 | 0.693 |
| 0.25 | −1.386 | 1.386 |
| 1/27 ≈ 0.037 | −3.296 | 3.296 |
| 0.001 | −6.908 | 6.908 |
| 0 | −∞ | ∞ |

Read the last column as **surprise**. A right answer predicted with certainty costs 0. Halving the probability adds 0.693 every time. A probability of 0 costs infinity: the model said the right answer was impossible.

Two more properties make logs the right tool. The log of a product is the sum of the logs, so the probability of a whole name (a product of small numbers) becomes a sum, which computers handle without underflowing to 0. And confident mistakes cost a lot: one prediction at 0.001 costs as much as ten at 0.5.

In PyTorch, `torch.log(t)` or `t.log()` takes the natural log of every number in a tensor.

## The loss: average negative log-likelihood {#nll}
> Take the log of the probability given to each right answer, average them, and flip the sign. Lower is better, and 0 would mean perfect certainty every time.

This score is the **negative log-likelihood** (NLL), and it's the loss for every model in this course:

1. For each prediction, take **ln of the probability** given to the right answer. Every one is 0 or negative.
2. **Average** them.
3. **Flip the sign**, so the result is positive and smaller means better.

Worked example with two predictions, at 0.5 and 0.25:

```
ln 0.5  = −0.693
ln 0.25 = −1.386
average = −1.040
loss    =  1.040 nats
```

The unit is **nats**, short for "natural units", because the log is the natural log. With log base 2 you'd get bits instead, a different number for the same model, so the course always uses ln.

The challenge's tests catch the classic mistakes:

| Mistake | What goes wrong |
|---|---|
| Sum instead of average | The loss grows with the amount of data, so twice the names looks twice as bad. |
| Forgetting the minus sign | The loss comes out negative, and "lower is better" breaks. |
| `log2` or `log10` | Bits or another unit; numbers don't match the course's. |
| Log of the average probability | For 0.5 and 0.25: −ln 0.375 = 0.981, not 1.040. One confident miss hides behind the good guesses. |

## The uniform baseline: ln 27 {#baseline}
> A model that gives all 27 symbols probability 1/27 scores exactly ln 27 ≈ 3.2958 nats, whatever the data. Every real model must get below that.

If every prediction gives the right answer probability 1/27, every term in the average is the same:

```
−ln(1/27) = ln 27 ≈ 3.2958
```

The average of identical numbers is that number, so the loss is 3.2958 for 5 names or 32,033. That makes it a clean reference point:

| Loss | What it tells you |
|---|---|
| far above 3.30 | The model is confidently wrong. A fresh network often starts like this; 0.4's cards show 27.9, and 0.4 fixes it. |
| about 3.30 | The model knows nothing, but is honest about it. A well-initialised network starts here. |
| below 3.30 | The model has learned something. The bigram in 0.3 gets to about 2.45. |

Your quality metric for this lesson is the baseline computed over all of names.txt on your own machine. It must come out between 3.29 and 3.30. If it does, your data loading and your loss both work.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenge and metrics.

- **GPU**: a processor with thousands of simple cores, built for the parallel arithmetic that neural networks need.
- **CUDA**: NVIDIA's software layer for running code on its GPUs. A "CUDA build" of PyTorch can use the GPU.
- **Matmul**: matrix multiplication, `a @ b`. Each output cell is a dot product of a row and a column.
- **FLOP, FLOP/s, TFLOP/s**: one floating-point operation; operations per second; 10¹² operations per second.
- **fp32, fp16, bf16**: 32-bit floats, and two 16-bit formats. bf16 keeps fp32's range with fewer digits.
- **Tensor cores**: GPU circuits for fast 16-bit matmuls.
- **Asynchronous**: the call returns before the work is done. `torch.cuda.synchronize()` waits for it.
- **Warm-up**: untimed runs before timing, so one-off start-up costs don't count.
- **Probability**: a number from 0 to 1. The probabilities of all outcomes sum to 1.
- **ln (natural log)**: turns a probability into a log-probability: 0 for certainty, more negative for less likely.
- **Negative log-likelihood (NLL)**: the average of −ln p over the right answers. The loss for every model in the course.
- **Nats**: the unit of a loss computed with the natural log. Bits use log base 2.
- **Baseline**: the score of a simple reference model. Here, the uniform model's ln 27 ≈ 3.2958.
- **Uniform model**: gives every symbol the same probability, 1/27.
- **names.txt**: Karpathy's list of 32,033 names, 228,146 predictions in total, used through phase 0.
- **Illustrative replay**: example output written for the course to show what healthy looks like. Not a recording.
