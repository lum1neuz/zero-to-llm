---
gist: You'll learn a few more Python tools (dictionaries and some handy shortcuts), then meet tensors properly. Tensors are grids of numbers that do math on a whole grid at once.
minutes: 20
---

## What this lesson is for {#why}
> This lesson gets you from "I can write a small loop" to reading the code of the next five lessons comfortably.

In the Start here lessons you wrote loops, worked with grids, and computed a loss. This lesson adds the tools that the real course code uses all the time:

1. **Dictionaries**: a lookup table from one thing to another, like letters to numbers.
2. **A few Python shortcuts** that you'll see everywhere.
3. **Tensors**: grids of numbers you can do math on in one step.
4. **Broadcasting**: how a tensor stretches a small grid to fit a bigger one.

Every exercise is a small piece of **makemore**, the name-inventing model you'll finish in lesson 0.3.

Take your time. Each section is short, and every idea has an example you can check by hand.

## What you need from before {#before}
> From Start here: lists, loops, functions, grids with rows and columns, and percentages.

You'll lean on these:

- Lists like `["a", "b"]`, with positions from 0.
- `for x in things:` loops, and `if`.
- Functions with `def` and `return`.
- A grid as a list of rows, and `grid[row][col]`.
- Turning a row of counts into chances by dividing by the row's total.
- A first look at tensors and their **shape**, from W3.

## Dictionaries: a lookup table {#dicts}
> A dictionary maps keys to values. `d[key]` looks a value up.

A model can't read letters, only numbers (W1). So we need a lookup table: letter in, number out. In Python that's a **dictionary**:

```python
stoi = {"a": 1, "b": 2, "c": 3}
stoi["b"]    # 2
```

- Curly brackets `{ }` make a dictionary.
- Each entry is **key: value**. Here the keys are letters and the values are numbers.
- `stoi["b"]` looks up the key `"b"` and gives back `2`.

The name `stoi` is short for "string to integer". We also keep the opposite table, `itos` ("integer to string"), to turn the model's numbers back into letters:

```python
itos = {1: "a", 2: "b", 3: "c"}
itos[2]      # "b"
```

You can add an entry at any time:

```python
stoi["."] = 0
```

Now `"."` maps to 0. Makemore uses `"."` to mark where a name starts and ends.

## Shortcut: a loop that builds a list {#comprehension}
> A list comprehension builds a whole list in one line. Read it as "this, for each item in that".

You could turn a word into numbers with a loop:

```python
nums = []
for ch in "cab":
    nums.append(stoi[ch])
# nums is [3, 1, 2]
```

Python has a one-line shortcut that does the same thing:

```python
nums = [stoi[ch] for ch in "cab"]
```

Read it from the middle out: "for each `ch` in `"cab"`, take `stoi[ch]`, and put them all in a list."

It isn't magic, just a shorter way to write the loop above. When you see one, read it as that loop. (If you know Java: it's like a loop with `list.add`, or a stream with `map`.)

The same trick works for dictionaries, with `{key: value for …}`. You'll see that in the reorder card.

## Shortcut: counting along with enumerate {#enumerate}
> `enumerate(things)` gives you each item together with its position, so you don't need your own counter.

```python
for i, ch in enumerate(["a", "b", "c"]):
    print(i, ch)
```

prints

```
0 a
1 b
2 c
```

Each time round, you get two things: the position `i` and the item `ch`. That's handy for numbering letters. Adding 1 to the position gives a=1, b=2, c=3.

## Shortcut: slices {#slices}
> A slice copies part of a list or a word. `word[1:3]` starts at position 1 and stops just before 3.

```
position:  0    1    2    3    4
word   =  "e"  "m"  "m"  "a"  "."
```

| You write | You get | In words |
|---|---|---|
| `word[1:3]` | "mm" | positions 1 and 2 (stops before 3) |
| `word[:2]` | "em" | from the start, stop before 2 |
| `word[1:]` | "mma." | from position 1 to the end |
| `word[-1]` | "." | the last item |

The "stops just before" rule is the one people trip on. `word[1:3]` has 3 − 1 = 2 items.

## Shortcut: zip walks two lists together {#zip}
> `zip(a, b)` pairs items up: the first with the first, the second with the second. With a word and itself shifted by one, it gives every pair of neighbours.

```python
word = ".emma."
for a, b in zip(word, word[1:]):
    print(a, b)
```

`word[1:]` is the word without its first letter. Lining the two up gives each letter next to the one after it:

```
. e
e m
m m
m a
a .
```

These neighbour pairs are called **bigrams**. Makemore counts them across thousands of names, so it knows things like "after **e**, an **m** is fairly likely". That's the name model from W1, but for letters.

## Tensors: fast grids {#tensors}
> A tensor is a grid of numbers with a shape (its size) and a dtype (the kind of number). It does math on the whole grid at once.

In W3 you made a tensor from a list of rows. Here's another way to make one:

```python
import torch
x = torch.arange(12).reshape(3, 4)
```

- `torch.arange(12)` makes the numbers 0, 1, 2, … 11 in a row.
- `.reshape(3, 4)` folds them into 3 rows of 4.

```
        col0 col1 col2 col3
row 0 [  0,   1,   2,   3 ]
row 1 [  4,   5,   6,   7 ]
row 2 [  8,   9,  10,  11 ]
```

Every tensor has two labels you can print:

```python
print(x.shape, x.dtype)
```

- **shape** `[3, 4]`: 3 rows, 4 columns.
- **dtype** `int64`: the kind of number. `int64` means whole numbers. `float32` means numbers with decimals.

`print(t.shape, t.dtype)` is the most useful line in this whole course. When something goes wrong, the shape is usually why.

## Reading parts of a tensor {#indexing}
> Inside the square brackets, give one position per dimension, separated by a comma. `:` means "all of them".

With the same `x`:

| You write | In words | You get | Shape |
|---|---|---|---|
| `x[1, 2]` | row 1, column 2 | 6 | `[]` |
| `x[1]` | all of row 1 | [4, 5, 6, 7] | `[4]` |
| `x[:, 1]` | column 1 of every row | [1, 5, 9] | `[3]` |
| `x[-1]` | the last row | [8, 9, 10, 11] | `[4]` |

Notice the comma: tensors use `x[1, 2]` where a list of lists uses `grid[1][2]`.

The cards ask about one detail. A single number like `x[1]` gives a flat row, shape `[4]`. A slice like `x[1:2]` keeps it as a tiny grid with one row, shape `[1, 4]`.

**Many at once.** You can also give lists of positions: `P[[0, 2], [1, 0]]` picks `P[0, 1]` and `P[2, 0]` together. They're paired up like `zip`. One challenge uses this to look up many chances in one go.

## Adding along rows {#sum}
> `t.sum(dim=1)` adds across each row and gives one total per row, just like W3's challenge. `keepdim=True` keeps the totals as a column.

```python
N = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
```

| You write | You get | Shape | In words |
|---|---|---|---|
| `N.sum(dim=1)` | [6, 15] | `[2]` | one total per row |
| `N.sum(dim=1, keepdim=True)` | [[6], [15]] | `[2, 1]` | same, as a column |
| `N.sum(dim=0)` | [5, 7, 9] | `[3]` | one total per column |

What does `dim=1` mean? The shape is `[2, 3]`: dimension 0 is the rows (2 of them), dimension 1 is the columns (3 of them). `dim=1` adds **across the columns**, so the 3 disappears and you're left with one number per row.

`keepdim=True` leaves a 1 where the 3 was, giving shape `[2, 1]`, a single column. The next section shows why that matters.

## Broadcasting: dividing a whole grid at once {#broadcasting}
> When two tensors have different shapes, a dimension of size 1 gets stretched to fit. That lets you divide every row by its own total in one line.

In W3 you turned a row of counts into chances by dividing by the row's total. With tensors you can do every row at once:

```python
P = N / N.sum(dim=1, keepdim=True)
```

Here's what happens with the numbers:

```
N         [[1, 2, 3],
           [4, 5, 6]]

totals    [[ 6],          shape [2, 1]
           [15]]

stretched [[ 6,  6,  6],  now [2, 3]
           [15, 15, 15]]

N / that  [[1/6,  2/6,  3/6 ],
           [4/15, 5/15, 6/15]]
```

The column of totals was **stretched sideways** to match `N`. Then each number was divided by its own row's total. Every row now adds up to 1: a row of chances, like W3.

This stretching is called **broadcasting**. The rule: line the two shapes up from the right. Each pair of sizes must be equal, or one of them must be 1 (the 1 stretches).

```
[2, 3] with [2, 1]   ✓  1 stretches to 3
[2, 3] with    [3]   ✓  a missing size counts as 1
[2, 3] with    [2]   ✗  2 and 3 don't match
```

**The sneaky bug.** Without `keepdim=True` the totals have shape `[2]`, and here you'd get an error. That's good: you notice. But makemore's table is square, 27 by 27. Its row totals without keepdim have shape `[27]`, which lines up with the *columns*. Python happily divides every **column** by a row's total. You get wrong numbers and no error message. Keeping `keepdim=True` avoids it, which is why the cards keep asking about it.

## Whole numbers and decimals {#dtypes}
> Tensors of whole numbers can't be averaged. Turn them into decimals first with `.float()`.

Counts are whole numbers, so a tensor of counts has dtype `int64`. Asking for its average fails:

```python
counts = torch.tensor([1, 2, 4])
counts.mean()
# error: can't average whole numbers
counts.float().mean()
# tensor(2.3333)
```

`.float()` turns the tensor into decimals first. Dividing with `/` is fine either way: it always gives decimals.

## Setting up Python on your computer {#setup}
> The browser runs this lesson's challenges. The later GPU lessons need Python on your own machine, so set it up now. Step by step.

1. **Install Python 3.12** from python.org. On Windows, tick **"Add python.exe to PATH"** in the installer.
2. **Make a folder for the course** and open a terminal in it.
3. **Make a virtual environment**: a private set of Python libraries just for this course.
   ```bash
   python -m venv .venv
   ```
4. **Turn it on.** On Windows run `.venv\Scripts\activate`. On Mac or Linux run `source .venv/bin/activate`. Your terminal line now starts with `(.venv)`.
5. **Install PyTorch.** Go to pytorch.org, choose your system, and copy the install command it shows. If you have an NVIDIA graphics card, pick your CUDA version. Then also run `pip install numpy`.
6. **Check it works:**
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```
   If you see a version number, you're done.

If a step fails, don't worry. Lesson 0.1 walks through checking the setup in more detail.

## Words you'll see {#glossary}
> The new words from this lesson.

- **Dictionary**: a lookup table of `key: value` pairs. `d[key]` gives the value.
- **stoi / itos**: the tables from letters to numbers ("string to integer") and back.
- **Vocabulary**: all the symbols a model knows. For makemore that's a–z plus ".".
- **List comprehension**: `[thing for item in list]`, a one-line loop that builds a list.
- **enumerate**: gives each item together with its position.
- **Slice**: part of a list or word, like `word[1:3]`. It stops just before the end number.
- **zip**: pairs up items from two lists.
- **Bigram**: a pair of neighbouring letters, like "e" then "m".
- **Tensor**: a grid of numbers you can do math on all at once.
- **Shape**: a tensor's size, like `[3, 4]` (rows, then columns).
- **dtype**: the kind of number: `int64` (whole) or `float32` (decimals).
- **Dimension (dim)**: one direction of a tensor. dim 0 is the rows, dim 1 the columns.
- **keepdim**: keep a summed dimension as size 1, so the result is still a column.
- **Broadcasting**: stretching a size-1 dimension so two shapes fit together.
- **Virtual environment**: a private folder of Python libraries for one project.
