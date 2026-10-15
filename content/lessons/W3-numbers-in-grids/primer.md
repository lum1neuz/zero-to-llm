---
gist: Models keep their numbers in grids of rows and columns, like a spreadsheet. Adding up a row and turning a row into percentages are the two moves you'll use again and again.
minutes: 12
---

## What this lesson is for {#why}
> Language models store everything they learn in grids of numbers. This lesson gets you comfortable reading and working with those grids.

In W1 you saw a tiny model that counted which word comes after "good". A real model keeps lots of counts like that, one row for each starting letter or word. The natural shape for all those counts is a **grid**: rows going across, columns going down.

This lesson teaches three things:

1. How to point at one number in a grid.
2. How to add up a row.
3. How to turn a row of counts into percentages.

In lesson 0.3 you'll do exactly these three things to build a model that invents names.

## What you need from before {#before}
> From W2: lists, loops, sum, and functions with return.

You'll use these from W2:

- A **list** like `[2, 1, 1]`, with positions counted from 0.
- A **loop**: `for row in grid:` visits each item.
- **`sum(...)`** adds up a list, and **`len(...)`** counts its items.
- **`.append(x)`** adds `x` to the end of a list.
- A **function** gives back its answer with `return`.

## Rows and columns {#grid}
> A grid is a list of rows. Each row is a list of numbers.

Here's a small grid of counts:

| | col 0 | col 1 | col 2 |
|---|---|---|---|
| **row 0** | 2 | 1 | 1 |
| **row 1** | 0 | 3 | 1 |

In Python, you write it as a list whose items are lists:

```python
grid = [[2, 1, 1],
        [0, 3, 1]]
```

The outer brackets hold the whole grid. Each inner pair of brackets is one row. Lining the rows up underneath each other is just to make it readable. Python doesn't need it.

This grid has 2 rows and 3 columns. People write that as **2 × 3** ("two by three"), always rows first.

## Pointing at one number {#indexing}
> `grid[row][column]` picks one number. Both count from 0.

Getting one number takes two steps:

```python
grid[1]       # row 1: [0, 3, 1]
grid[1][1]    # item 1 of that row: 3
```

So `grid[1][1]` means "row 1, then column 1", which is **3**.

| You write | Row | Column | Value |
|---|---|---|---|
| `grid[0][0]` | 0 | 0 | 2 |
| `grid[0][2]` | 0 | 2 | 1 |
| `grid[1][1]` | 1 | 1 | 3 |

Tip: when you're unsure, say it out loud. "Row first, then column."

## Adding up a row {#row-total}
> One row's total is `sum(row)`. To get every row's total, loop over the rows.

Adding up one row is one step:

```python
sum([2, 1, 1])    # 4
```

To get the total of *every* row, you visit each row with a loop and collect its total in a new list:

| Row | Its total |
|---|---|
| [2, 1, 1] | 4 |
| [0, 3, 1] | 4 |

The result is a list with one number per row: `[4, 4]`. That's your first challenge. The loop is already written. You add each row's total to the list.

**Rows, not columns.** If you added *downwards* instead, you'd get column totals: 2 + 0, 1 + 3 and 1 + 1, which is `[2, 4, 2]`. Different numbers, and a different length. The tests will tell you if you've done that.

## From counts to percentages {#percent}
> A percent is a part divided by the total, times 100. A whole row of percentages adds up to 100.

Say the row `[5, 3, 2]` counts what came after "good": "morning" 5 times, "night" 3 times, "luck" 2 times. The total is 10.

| Count | ÷ total | × 100 | Percent |
|---|---|---|---|
| 5 | 0.5 | 50 | 50% |
| 3 | 0.3 | 30 | 30% |
| 2 | 0.2 | 20 | 20% |

```chart
kind: bar
title: What came after "good", counted
x: { label: next word, values: [morning, night, luck] }
y: { label: times seen, min: 0, max: 10 }
series:
  - { name: count, values: [5, 3, 2] }
caption: The raw counts out of a total of 10.
```

```chart
kind: bar
title: The same row as percentages
x: { label: next word, values: [morning, night, luck] }
y: { label: percent, min: 0, max: 100 }
series:
  - { name: percent, values: [50, 30, 20] }
caption: "Same shape, new scale: each count divided by the total of 10, times 100, so the bars add up to 100%."
```

Two things to notice:

- **Divide by the total of the row**, not by 100 and not by how many numbers there are.
- **They add up to 100%.** Every count is part of the same total, so together they make the whole.

The middle column, 0.5, 0.3 and 0.2, are **fractions**. Models usually keep them as fractions rather than percentages, and call them **probabilities**. 0.5 and 50% mean the same thing.

## Why percentages matter {#why-percent}
> Percentages let you compare rows with different totals, and they are exactly the model's guesses.

Imagine one row came from 10 messages and another from 1,000. Raw counts can't be compared fairly, but percentages can: 50% means "half the time" either way.

And a row of percentages is a model's whole prediction. "After *good*: 50% morning, 30% night, 20% luck." When the model writes, it picks the next word using these chances.

## A first look at tensors {#tensors}
> A tensor is a grid the computer can do math on very fast. From lesson 0.0 onwards, grids live in tensors.

So far your grid is a list of lists, and you used a loop to add up the rows. That works, but it's slow for big grids. Machine learning uses **tensors** instead: grids stored in a way the computer can process all at once.

```python
import torch
t = torch.tensor([[2, 1, 1],
                  [0, 3, 1]])
print(t.shape)
```

This prints `torch.Size([2, 3])`. A tensor's **shape** is its size: 2 rows, 3 columns.

With a tensor, every row's total takes one line: `t.sum(dim=1)` gives `[4, 4]`. You don't need to understand `dim=1` yet. Lesson 0.0 explains it slowly. For now, just know that the loop you write here and that one line do the same job.

## Common slips {#slips}
> Most grid mistakes come from mixing up rows and columns, or dividing by the wrong number.

| Slip | What happens | Fix |
|---|---|---|
| Swapped row and column | You get the wrong cell | Row first, then column |
| Added downwards | You get column totals | Loop over rows and `sum(row)` |
| Appended the row itself | You get lists back, not totals | Append `sum(row)` |
| Divided by 100 | The numbers are much too small | Divide by the row's total |
| Forgot × 100 | You get 0.5 instead of 50 | Multiply the fraction by 100 |

## Words you'll see {#glossary}
> The new words from this lesson.

- **Grid**: numbers arranged in rows and columns. In Python, a list of lists.
- **Row**: a line going across. **Column**: a line going down.
- **Index**: a position, counted from 0. `grid[r][c]` uses two indexes.
- **Total**: everything in a row added up.
- **Percent**: a part out of 100. 50% is a half.
- **Fraction / probability**: the same idea written as a number from 0 to 1. 0.5 is a half.
- **Tensor**: a grid the computer can do fast math on.
- **Shape**: a tensor's size, rows first: `[2, 3]` is 2 rows and 3 columns.
