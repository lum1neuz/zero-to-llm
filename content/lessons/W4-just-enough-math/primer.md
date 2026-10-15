---
gist: Four small math ideas carry most of this course. Chances multiply, powers repeat a multiplication, logs count the repeats, and "surprise" (minus the log of a chance) measures how wrong a model was.
minutes: 14
---

## What this lesson is for {#why}
> Models are judged by one number, the loss. This lesson builds up to it from everyday math, one step at a time.

When you train a model, the computer prints a number called the **loss** after every step. If it goes down, the model is learning. If it goes up, something's wrong.

The loss sounds technical, but it's built from four simple ideas:

1. **Chances** (probabilities), like "50% chance of heads".
2. **Multiplying chances** for things that happen in a row.
3. **Powers and logarithms**, which tame very big and very small numbers.
4. **Surprise**: how unexpected the right answer was to the model.

The loss is just the average surprise. By the end of this lesson you'll compute one yourself.

## What you need from before {#before}
> From W1: a model gives every possible next piece a chance. From W2–W3: averages and percentages.

- A **probability** (chance) is a number from 0 to 1. 0.5 is the same as 50%.
- A model's guess is a **row of chances** that add up to 1. For example: "morning" 0.5, "night" 0.3, "luck" 0.2.
- An **average** is the total divided by how many there are.

## Chances in a row multiply {#multiply}
> The chance of two things both happening, one after the other, is their chances multiplied.

Flip a coin. The chance of heads is 0.5.

Flip it twice. What's the chance of heads *both* times?

- Half the time the first flip is heads.
- Of those times, half the time the second is heads too.
- Half of a half is a quarter: **0.5 × 0.5 = 0.25**.

A model writing text works the same way. If it gives "h" a chance of 0.5, and then "i" (after "h") a chance of 0.2, the chance of writing "hi" is

```
0.5 × 0.2 = 0.1
```

That's 10%.

## The problem: tiny numbers {#tiny}
> Multiply many chances together and the result gets so small it's hard to work with.

A sentence has many pieces. Multiply 20 chances of 0.1 together and you get

```
0.1 × 0.1 × … (20 times)
= 0.00000000000000000001
```

That's a 1 with 20 zeros in front. Numbers like this are hard to read, hard to compare, and computers start losing accuracy with them. We need a better way to talk about them. That's what logarithms are for.

## Powers: multiplying again and again {#powers}
> A power is a short way to write "multiply this number by itself several times".

```
2 × 2 × 2      = 8     written 2³
10 × 10        = 100   written 10²
10 × 10 × 10   = 1000  written 10³
```

The small raised number counts how many times you multiplied. 2³ is "2, multiplied three times".

## Logarithms: counting the multiplications {#logs}
> A logarithm undoes a power. It answers "how many times did I multiply to get here?"

| Question | Written | Answer |
|---|---|---|
| How many 2s make 8? | log₂(8) | 3 |
| How many 10s make 1000? | log₁₀(1000) | 3 |
| How many 10s make 10? | log₁₀(10) | 1 |
| How many 10s make 1? | log₁₀(1) | 0 |

The last row is worth remembering: **the log of 1 is 0**. You don't need to multiply at all to get 1.

What about numbers *below* 1, like chances? Dividing is the opposite of multiplying, so their logs are **negative**:

| Number | log₁₀ |
|---|---|
| 1 | 0 |
| 0.1 | −1 |
| 0.01 | −2 |
| 0.00000000000000000001 | −20 |

That tiny number from before is now just −20. That's why logs help.

## The natural log {#ln}
> Python's `math.log` uses a special base, about 2.718. You don't need to know why, just how to use it.

Logs can count in 2s (log₂) or 10s (log₁₀). Machine learning almost always uses a third base, a number called **e**, which is about 2.718. That's the **natural log**, and it's what `math.log` in Python gives you.

You never need to work it out by hand:

```python
import math
math.log(1)      # 0.0
math.log(0.5)    # -0.693...
math.log(0.1)    # -2.302...
```

The same rules hold: the log of 1 is 0, and numbers below 1 give negative logs.

## Surprise: measuring how wrong a guess was {#surprise}
> Surprise is minus the log of the chance the model gave to what actually happened. Certain and right costs 0. Unlikely and right costs a lot.

Say the model guessed the next letter, and the real next letter was "a". Look at the chance the model had given to "a":

| Chance it gave "a" | −log(chance) = surprise |
|---|---|
| 1.0 | 0.00 |
| 0.5 | 0.69 |
| 0.25 | 1.39 |
| 0.1 | 2.30 |
| 0.01 | 4.61 |

```chart
kind: line
title: Surprise against the chance given to "a"
x: { label: chance the model gave "a", values: [0.01, 0.1, 0.25, 0.5, 1.0] }
y: { label: surprise, min: 0, max: 5 }
series:
  - { name: surprise, values: [4.61, 2.30, 1.39, 0.69, 0.00] }
highlight: { x: 0.5, label: "half a chance, 0.69" }
caption: Surprise is 0 when the model was certain and shoots up as the chance it gave the right answer falls toward 0.
```

Read the table and the chart as a story:

- The model was **certain** and right: no surprise at all.
- It gave the right answer **half** a chance: a little surprise.
- It thought the right answer was **very unlikely**: big surprise.

We put a **minus** in front because the logs of chances are negative, and we want surprise to be a positive number where bigger means worse.

This is your first challenge: write `surprise(p)` in one line.

## The loss: average surprise {#loss}
> A model's loss is its average surprise over many guesses. Training is the job of making it smaller.

One guess is not enough to judge a model. So we look at many guesses and average their surprises:

| Guess | Chance given to the right answer | Surprise |
|---|---|---|
| 1 | 0.5 | 0.69 |
| 2 | 0.25 | 1.39 |
| 3 | 1.0 | 0.00 |
| | **Average (the loss)** | **0.69** |

(0.69 + 1.39 + 0.00) ÷ 3 = 0.69.

That's the **loss**. In lesson 0.3 it has a longer name, **negative log likelihood**, but it's exactly this. Lower is better:

- A loss near **0** means the model gave the right answers chances close to 1.
- A higher loss means it was often surprised.

Your second challenge computes this average using the `surprise` function you wrote.

## Why logs instead of chances? {#why-logs}
> Logs turn "multiply many chances" into "add many surprises", which is easier and doesn't create tiny numbers.

Remember the chance of "hi": 0.5 × 0.2 = 0.1. Look at the surprises:

```
surprise(0.5) + surprise(0.2)
= 0.69 + 1.61
= 2.30
```

And surprise(0.1) is also **2.30**. Multiplying chances is the same as adding surprises. That's the magic of logs.

So instead of multiplying thousands of chances into an impossibly tiny number, a model adds up thousands of ordinary-sized surprises. Then it averages them so long and short texts can be compared fairly.

## Words you'll see {#glossary}
> The new words from this lesson.

- **Probability / chance**: how likely something is, from 0 (never) to 1 (always).
- **Power**: repeated multiplying. 2³ = 2 × 2 × 2 = 8.
- **Logarithm (log)**: counts the multiplications. log₂(8) = 3. The log of 1 is 0. Numbers below 1 have negative logs.
- **Natural log**: the log Python's `math.log` uses (base e ≈ 2.718).
- **Surprise**: −log(chance the model gave to what really happened).
- **Loss**: the average surprise over many guesses. Lower means better guesses.
- **Negative log likelihood (NLL)**: the formal name for this loss. You'll meet it in lesson 0.3.
