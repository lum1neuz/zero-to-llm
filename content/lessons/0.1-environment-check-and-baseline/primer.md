---
gist: Before you train anything, you check two numbers. One tells you your graphics card is really doing the work. The other is the score of a model that knows nothing, about 3.30, which every later model has to beat.
minutes: 20
---

## Start here {#why}
> This lesson does two checks before any training: is your graphics card doing the work, and what score does "knowing nothing" get?

Before a long run, a runner checks two things. Are my shoes on properly? And how fast is walking pace, so I know whether I'm really running?

This lesson does the same for training models.

**Check 1: is the graphics card doing the work?** Training a model means doing a huge amount of arithmetic. A graphics card does that arithmetic much faster than the computer's main chip. You'll time one big calculation and see how fast your card is.

**Check 2: what's "walking pace" for a model?** In 0.3 you'll build a model that guesses the next letter of a name. Before it learns anything, the fairest guess is to give every letter the same chance. You'll work out the score of that fair guess. It's called the **baseline**. A model that has learned something must score better than it.

The browser challenge does check 2 with a few names. Check 1 needs your own graphics card, so you run a short script on your computer. The lesson shows example output so you know what to expect.

## What you need from before {#before}
> You need lists, tensors and `.mean()` from 0.0, and probabilities and logs from Start here.

From the Start here lessons and 0.0:

| Idea | Reminder |
|---|---|
| Probability | how likely something is, from 0 (never) to 1 (always). 0.25 means 25%. |
| Logarithm | from W4. You'll meet the natural log, ln, again below. |
| Tensor | a grid of numbers, like a list of lists, with one shape. |
| `.mean()` | the average of all the numbers in a tensor. |
| `.item()` | turns a one-number tensor into a plain Python number. |

A quick reminder of `.mean()` and `.item()`:

```python
p = torch.tensor([0.5, 0.25])
avg = p.mean()
print(avg)
print(avg.item())
```

| Line | What it does |
|---|---|
| `p = torch.tensor(...)` | makes a tensor holding 0.5 and 0.25 |
| `avg = p.mean()` | averages them: (0.5 + 0.25) / 2 |
| `print(avg)` | prints `tensor(0.3750)` |
| `print(avg.item())` | prints the plain number `0.375` |

## Two kinds of chip {#gpu}
> The computer's main chip (CPU) has a few fast workers. A graphics card (GPU) has thousands of simple ones, which is perfect for the arithmetic models need.

Your computer has a main chip, the **CPU**. It's very good at doing one complicated thing after another.

Many computers also have a graphics card, the **GPU**. It was built to colour millions of screen pixels at once. So it has thousands of small, simple workers that all work at the same time.

Think of marking 10,000 multiple-choice tests. One expert marks them one after another. A hall of 2,000 students, each marking five tests, finishes far sooner, even though each student is slower than the expert.

Models are like the stack of tests: lots of small, separate pieces of arithmetic. That's why the GPU wins.

**CUDA** is the software that lets programs use NVIDIA graphics cards. PyTorch needs a version built with CUDA to use your card. In 0.0 you checked this with `torch.cuda.is_available()`.

## Multiplying two grids {#matmul}
> To multiply two grids, each cell of the answer comes from one row of the first grid and one column of the second: multiply the pairs, then add.

Most of a model's work is one operation: **matrix multiplication**, or **matmul** for short. A matrix is a grid of numbers. In PyTorch you write it with `@`.

Here's how one cell of the answer is made. Take a row from the first grid and a column from the second. Multiply them in pairs and add the results. This is called a **dot product**.

```
a = [[1, 2],     b = [[5, 6],
     [3, 4]]          [7, 8]]
```

| Answer cell | Row of a | Column of b | Arithmetic | Result |
|---|---|---|---|---|
| top left | 1, 2 | 5, 7 | 1·5 + 2·7 | 19 |
| top right | 1, 2 | 6, 8 | 1·6 + 2·8 | 22 |
| bottom left | 3, 4 | 5, 7 | 3·5 + 4·7 | 43 |
| bottom right | 3, 4 | 6, 8 | 3·6 + 4·8 | 50 |

In code:

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
print(a @ b)
```

This prints `[[19, 22], [43, 50]]`. Notice that no cell needs another cell's answer. All four could be worked out at the same moment, by four different workers. That's exactly the kind of job a GPU is good at.

## Counting the work {#flops}
> Multiplying two n-by-n grids takes about 2 × n × n × n steps of arithmetic. For the big test grid that's about 137 billion.

One multiply or one add is called a **FLOP**, short for "floating-point operation". (A floating-point number is just a number with a decimal point.)

Let's count the FLOPs for multiplying two square grids that are n numbers wide.

1. **One answer cell** multiplies n pairs and adds them up. That's about n multiplies and n adds: **2 × n** FLOPs.
2. **The answer** has n rows and n columns: **n × n** cells.
3. **In total**: 2 × n × n × n FLOPs. In short, 2n³.

Check it on the 2 × 2 example. Each cell had 2 multiplies and 1 add. The rule says 2 × 2 × 2 × 2 = 16, and the real count is 12. The rule rounds the adds up slightly, which doesn't matter for big grids.

The lesson's test uses grids 4096 numbers wide:

```
2 × 4096 × 4096 × 4096
  = 137,438,953,472
  ≈ 137 billion FLOPs
```

Speed is FLOPs per second. The numbers are huge, so we count in **trillions per second**, written **TFLOP/s**. (Tera means a trillion.)

To get it, divide the FLOPs by the seconds taken, then by a trillion. Say one multiplication took 2.01 milliseconds. A millisecond is a thousandth of a second, so that's 0.00201 seconds.

```
137 billion / 0.00201 seconds
  ≈ 68.4 trillion per second
  = 68.4 TFLOP/s
```

The easy mistake is the milliseconds. Divide by 0.00201, not by 2.01.

## Shorter numbers are faster {#dtypes}
> Numbers can be stored with 4 bytes or with 2. The 2-byte formats keep fewer digits, but the graphics card has special parts that make them about three times faster.

A computer stores each number in a fixed amount of memory. A **byte** is a small unit of memory.

| Format | Bytes per number | Keeps about | Used for |
|---|---|---|---|
| `float32` (fp32) | 4 | 7 digits | the normal choice; safe everywhere |
| `float16` (fp16) | 2 | 3 digits, and only numbers up to 65,504 | fast, but needs care |
| `bfloat16` (bf16) | 2 | 2–3 digits, but huge numbers are fine | fast training on newer cards |

Newer NVIDIA cards have **tensor cores**: parts of the chip built only for multiplying grids of 2-byte numbers. With them, fp16 and bf16 run about three times faster than fp32.

Here are example timings for one 4096-wide multiplication. They're illustrative numbers for an RTX 3090 card, written for the course:

| Where | Format | Time | TFLOP/s |
|---|---|---|---|
| CPU | fp32 | 182.40 ms | 0.8 |
| GPU | fp32 | 6.21 ms | 22.1 |
| GPU | fp16 | 2.01 ms | 68.4 |
| GPU | bf16 | 2.03 ms | 67.7 |

Your card will give different numbers. Look for the pattern instead: the GPU is many times faster than the CPU, and the 2-byte formats are a few times faster than fp32. Older cards, like the T4 on free Google Colab, have no bf16 tensor cores. Use fp16 there.

## Timing a graphics card fairly {#timing}
> The graphics card works in the background, so you must wait for it to finish before you stop the clock. And the first run is slow, so you skip it.

When Python tells the GPU "multiply these", it doesn't wait for the answer. It hands over the job and moves straight on.

It's like ordering at a restaurant. The waiter takes your order and walks away at once. Your food isn't ready yet. If you timed "ordering", you'd think the kitchen cooks in one second.

The cards show this trap: a timing of 0.00004 seconds. That would mean 3,400 TFLOP/s, which is impossible.

Two rules make the timing fair:

1. **Wait for the food.** `torch.cuda.synchronize()` waits until the GPU has finished everything it was given. Call it before you start the clock and again before you stop it.
2. **Let the kitchen warm up.** The very first GPU job in a program is slow, because the card is getting set up. Run the multiplication a few times without timing it. Then time many runs and take the average.

## The check you run on your computer {#env-check}
> You run one short script. It prints your versions, your card's name, a table of timings, and the baseline score over all the names.

The script is in the lesson's "Check your GPU" step, under "Run it yourself". It needs a file called `names.txt`: 32,033 real first names. The script has the download link.

It does three things:

1. Prints your PyTorch and CUDA versions and your card's name. If PyTorch can't see a card, it stops and says so.
2. Times the 4096-wide multiplication: once on the CPU, then on the GPU in fp32, fp16 and bf16.
3. Works out the baseline score over every name in the file.

The lesson plays an **illustrative** example of the output. It was written for the course to show what healthy output looks like. It isn't a recording of a real run.

Healthy output shows your card's name, a few milliseconds per GPU row, fp16 and bf16 a few times faster than fp32, and a baseline of 3.2958.

| If you see | It probably means |
|---|---|
| "No GPU" | You installed the CPU-only PyTorch. `torch.version.cuda` prints `None`. Reinstall using the command from pytorch.org. |
| GPU rows as slow as the CPU row | The work isn't really on the GPU, or the timing skips `synchronize()`. |
| Numbers that jump around | A laptop card heating up and slowing down. Run more repeats. |

You'll record your best fp16 or bf16 speed. Anything above 5 TFLOP/s passes. A CPU manages about 1.

## Scoring one guess {#surprise}
> For each letter, the model gives every possible next letter a chance. We look at the chance it gave the right one, and turn that into a "surprise" score with the log.

Makemore's model can choose from 27 symbols: the letters a to z, plus "." which marks the start and end of a name. Each time, it gives all 27 a chance, and the chances add up to 1.

The name `emma`, written with its markers as `.emma.`, asks for five guesses:

| After | The right answer is |
|---|---|
| "." | e |
| e | m |
| m | m |
| m | a |
| a | "." |

So a name with 4 letters gives 5 guesses. For each guess, we only care about one number: the chance the model gave to the right answer.

Next we turn that chance into a **surprise** score. W4 introduced logarithms. Here we use the **natural log**, written **ln**, and we flip its sign:

| Chance of the right answer | ln | Surprise, −ln |
|---|---|---|
| 1 (certain) | 0 | 0 |
| 0.9 | −0.105 | 0.105 |
| 0.5 | −0.693 | 0.693 |
| 0.25 | −1.386 | 1.386 |
| 1/27 (about 0.037) | −3.296 | 3.296 |
| 0.001 | −6.908 | 6.908 |
| 0 (impossible) | minus infinity | infinity |

Read the last column like this. Being sure and right costs nothing. Every time the chance halves, the surprise goes up by 0.693. And saying the right answer was impossible costs infinity.

That's why logs are used: a confident wrong guess is punished hard. One guess at 0.001 costs as much as ten guesses at 0.5.

In PyTorch, `t.log()` takes the natural log of every number in a tensor.

## The loss: the average surprise {#nll}
> The loss is the average surprise over all the guesses. Lower is better. Its full name is the negative log-likelihood.

Here's the recipe in words:

1. For each guess, take the **ln** of the chance given to the right answer.
2. **Average** those numbers.
3. **Flip the sign**, so the result is positive.

This is the **loss** every model in this course uses. Its proper name is the **negative log-likelihood**, or **NLL**. "Likelihood" just means the chances given to what really happened.

A worked example with two guesses. The model gave the right answers chances 0.5 and 0.25:

| Step | Value |
|---|---|
| ln 0.5 | −0.693 |
| ln 0.25 | −1.386 |
| average | −1.040 |
| flip the sign | **1.040** |

The unit is called **nats**, because we used the natural log. With log base 2 you'd get a different number for the same model, so the course always uses ln.

The challenge's tests catch these common mistakes:

| Mistake | What goes wrong |
|---|---|
| Adding up instead of averaging | Twice as many names would look twice as bad. |
| Forgetting to flip the sign | The loss is negative, and "lower is better" breaks. |
| Using `log2` or `log10` | A different unit, so the numbers don't match the course. |
| Averaging the chances first, then taking the log | For 0.5 and 0.25 you'd get 0.981, not 1.040. One bad guess hides behind the good ones. |

## The baseline: about 3.30 {#baseline}
> A model that gives all 27 symbols the same chance, 1/27, always scores ln 27, about 3.2958. Every real model has to beat that.

Imagine a model that knows nothing. To be fair, it gives every symbol the same chance: 1 in 27.

Then every guess has the same surprise: −ln(1/27), which is ln 27, about 3.2958. And the average of numbers that are all the same is just that number. So this model scores 3.2958, whether you test it on 5 names or 32,033.

That makes it a good yardstick:

| Loss | Meaning |
|---|---|
| far above 3.30 | The model is confidently wrong. A new model often starts like this. Lesson 0.4 shows a start of 27.9 and fixes it. |
| about 3.30 | The model knows nothing, but admits it. A well-prepared model starts here. |
| below 3.30 | The model has learned something. In 0.3 the letter-pair model reaches about 2.45. |

Your score for this lesson is the baseline worked out over all of `names.txt` on your own computer. It should land between 3.29 and 3.30. If it does, you've loaded the names and computed the loss correctly.

## Words you'll see {#glossary}
> A short list of the new words in this lesson's cards, challenge and scores.

- **CPU**: the computer's main chip. A few fast workers.
- **GPU (graphics card)**: a chip with thousands of simple workers, great for the arithmetic models need.
- **CUDA**: NVIDIA's software that lets programs use its graphics cards.
- **Matmul (matrix multiplication)**: multiplying two grids with `@`. Each answer cell is a row times a column.
- **Dot product**: multiply two lists in pairs, then add up the results.
- **FLOP**: one multiply or one add.
- **TFLOP/s**: trillions of FLOPs per second.
- **fp32, fp16, bf16**: ways of storing numbers in 4 bytes or 2 bytes.
- **Tensor cores**: parts of newer graphics cards built for fast 2-byte grid multiplications.
- **Synchronize**: wait until the graphics card has finished its work.
- **Warm-up**: a few untimed runs first, so setup time doesn't count.
- **Probability (chance)**: how likely something is, from 0 to 1.
- **ln (natural log)**: the log used for scores in this course.
- **Surprise**: −ln of the chance given to the right answer. 0 for a sure right guess.
- **Loss**: one number that says how bad a model's guesses are. Lower is better.
- **Negative log-likelihood (NLL)**: the average surprise. The loss used in this course.
- **Nats**: the unit of a loss worked out with ln.
- **Baseline**: the score of a simple model to compare against. Here, a model that gives every symbol a 1-in-27 chance: 3.2958.
- **names.txt**: a list of 32,033 first names, used throughout phase 0.
- **Illustrative**: example output written for the course to show what healthy looks like. Not a recording.
