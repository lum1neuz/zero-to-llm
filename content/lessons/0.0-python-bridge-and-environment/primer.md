---
gist: Python reads like Java with the braces and type names removed, and a tensor is a grid of numbers you do math on all at once. This lesson gets both into your hands.
minutes: 15
---

## What this lesson is for {#why}
> You'll learn just enough Python and tensors to read every line of code in the next five lessons, starting from Java and HTML/CSS.

The whole course is written in Python with a library called **PyTorch**. You don't have to master Python. You need to *read* it comfortably and write small pieces of it.

Every exercise here is a small piece of **makemore**, a program you finish in lesson 0.3. Makemore reads a list of real names like `emma`, `olivia` and `ava`. It learns which letter tends to follow which, then invents new names that sound real. To do that, a computer needs three things, and this lesson teaches each one:

1. **Turn letters into numbers.** Models only do arithmetic. That's Python dictionaries.
2. **Keep numbers in grids.** "How often does `b` follow `a`?" is a cell in a 27 × 27 table. That's a tensor.
3. **Do math on whole grids at once.** Turn a table of counts into probabilities in one line. That's broadcasting.

If something below doesn't click on the first read, keep going. The cards and challenges come back to every idea with an example.

## Python for a Java reader {#python}
> Same ideas as Java, less ceremony: no types on variables, no braces, no semicolons. Indentation decides what's inside a block.

Here's the same small function in both languages:

```java
// Java
static int countVowels(String word) {
    int count = 0;
    for (char ch : word.toCharArray()) {
        if ("aeiou".indexOf(ch) >= 0) {
            count++;
        }
    }
    return count;
}
```

```python
# Python
def count_vowels(word):
    count = 0
    for ch in word:
        if ch in "aeiou":
            count += 1
    return count
```

What changed:

| Java | Python | Note |
|---|---|---|
| `int count = 0;` | `count = 0` | No type, no semicolon. The value has a type, the variable doesn't. |
| `{ … }` | `:` then indent | The indentation *is* the block. Use 4 spaces. A wrong indent is a real error. |
| `for (char ch : word.toCharArray())` | `for ch in word:` | `for … in` walks any sequence: strings, lists, dictionaries. |
| `"aeiou".indexOf(ch) >= 0` | `ch in "aeiou"` | `in` asks "is it inside?" for strings, lists and dict keys. |
| `count++` | `count += 1` | Python has no `++`. |
| `null`, `true`, `false` | `None`, `True`, `False` | Capitalised. |
| `&&`, `\|\|`, `!` | `and`, `or`, `not` | Words, not symbols. |
| `System.out.println(x)` | `print(x)` | `print(a, b)` prints both with a space between. |
| `// comment` | `# comment` | |

One more Python habit: **functions can return several values**. `return q, scale` returns a pair, and `q, scale = quantize(w)` unpacks it into two variables. You'll see this a lot.

## Lists and slices {#slices}
> A Python list is an ArrayList with friendlier syntax. Slices like `s[1:3]` copy a range, and the end index is never included.

```python
# like List<String> in Java
names = ["emma", "olivia", "ava"]

names[0]     # "emma"
names[-1]    # "ava" (counts from the end)
len(names)   # 3, like names.size()
names.append("isabella")
```

**Slicing** copies part of a list (or a string) with `start:end`. The start is included and the end is **not**. This is called *half-open*:

```
index:   0   1   2   3   4
word  = "e" "m" "m" "a" "."

word[1:3]  →  "mm"
word[:2]   →  "em"
word[1:]   →  "mma."
word[-1]   →  "."
```

`word[1:3]` takes items 1 and 2 and stops *before* 3. `word[:2]` starts at the beginning, `word[1:]` runs to the end, and `word[-1]` is the last item.

Java's `substring(1, 3)` works the same way, so you already know the rule.

## Dictionaries, enumerate and comprehensions {#comprehensions}
> A dict is a HashMap you write with `{}`. A comprehension builds a whole list or dict in one line, like a Java stream with `map`.

A model can't do arithmetic on `"e"`, so every character gets a number. Two lookup tables do it:

```python
# "string to integer", like a
# HashMap<String, Integer>
stoi = {".": 0, "a": 1, "b": 2}

# "integer to string": the reverse
itos = {0: ".", 1: "a", 2: "b"}

stoi["a"]   # 1
```

You don't type all 27 entries by hand. A **comprehension** builds them:

```python
# every distinct letter, sorted:
# ['a', 'b', ..., 'z']
chars = sorted(set("".join(words)))

# {'a': 1, 'b': 2, ..., 'z': 26}
stoi = {ch: i + 1
        for i, ch in enumerate(chars)}

# "." marks a name's start and end
stoi["."] = 0

# flip every pair around
itos = {i: ch for ch, i in stoi.items()}
```

Read a comprehension from the middle: "for each `i, ch` in `enumerate(chars)`, make an entry `ch: i + 1`." In Java you'd write a loop that calls `map.put(ch, i + 1)`, or a stream with `Collectors.toMap`.

Some pieces you haven't met yet:

- **`set(...)`** removes duplicates, and **`sorted(...)`** returns a sorted list.
- **`"".join(words)`** glues a list of strings into one string, like `String.join("", words)`.
- **`enumerate(chars)`** gives `(0, 'a'), (1, 'b'), …`: the index and the item together, so you don't need a counter variable.
- **`stoi.items()`** gives the `(key, value)` pairs, like `map.entrySet()`.

Encoding a name is then one line: `[stoi[ch] for ch in "emma"]` gives `[5, 13, 13, 1]`.

## zip: walking two lists together {#zip}
> `zip(a, b)` pairs up items from two sequences. With a list and itself shifted by one, it gives every neighbouring pair: exactly what makemore counts.

```python
word = ".emma."
for a, b in zip(word, word[1:]):
    print(a, b)
```

prints

```
. e
e m
m m
m a
a .
```

`word[1:]` is the same string without its first letter, so `zip` lines up each letter with the one after it. These pairs are called **bigrams**. Makemore counts them across thousands of names: how often does `m` follow `e`? That count is the whole model in lesson 0.3.

## What a tensor is {#tensors}
> A tensor is a grid of numbers, like a Java `double[][]` but of any number of dimensions, with one shape and one number type, built for doing math on the whole grid at once.

```python
import torch
x = torch.arange(12).reshape(3, 4)
print(x)
print(x.shape, x.dtype)
```

```
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
torch.Size([3, 4]) torch.int64
```

- **`torch.arange(12)`** makes the numbers 0 to 11 in a row, and **`.reshape(3, 4)`** folds them into 3 rows of 4.
- **The shape** `[3, 4]` lists the size of each dimension: 3 rows, 4 columns. A single number has shape `[]`, a row of 5 numbers has shape `[5]`, and a stack of 32 such grids would be `[32, 3, 4]`.
- **The dtype** is the number type of *every* cell: `int64` for whole numbers (made from Python ints) or `float32` for decimals (made from Python floats). One tensor never mixes them.

Why not just use lists of lists? Speed. `x * 2` doubles all 12 numbers in one operation, and on a GPU it does millions at once. Every model in this course is tensors multiplied by tensors.

The most useful debugging line in the whole course is `print(t.shape, t.dtype)`. When something goes wrong, the shape is usually the reason.

## Reading parts of a tensor {#indexing}
> Index with one position per dimension, separated by commas. `:` means "all of this dimension". A plain number removes that dimension from the result.

Using the same `x`:

```
          col 0  col 1  col 2  col 3
row 0  [    0,     1,     2,     3 ]
row 1  [    4,     5,     6,     7 ]
row 2  [    8,     9,    10,    11 ]
```

| Code | Means | Result | Shape |
|---|---|---|---|
| `x[1, 2]` | row 1, column 2 | `6` | `[]` |
| `x[1]` | row 1, all columns | `[4, 5, 6, 7]` | `[4]` |
| `x[:, 1]` | all rows, column 1 | `[1, 5, 9]` | `[3]` |
| `x[-1]` | the last row | `[8, 9, 10, 11]` | `[4]` |
| `x[1:2]` | row 1, as a slice | `[[4, 5, 6, 7]]` | `[1, 4]` |

The last two rows of the table show the rule the cards quiz you on. `x[1]` (a number) gives shape `[4]`: the row dimension is gone. `x[1:2]` (a slice) gives `[1, 4]`: you still have a grid, it just has one row.

**Picking many cells at once.** You can index with *lists* of positions. `P[xs, ys]` with `xs = [0, 2]` and `ys = [1, 0]` returns `[P[0, 1], P[2, 0]]`. The positions are paired up like `zip`. One of the challenges uses this to look up the probability of every bigram in one go.

## Adding along a dimension {#sum}
> `t.sum(dim=1)` adds across each row and gives one total per row. `dim=0` adds down each column. `keepdim=True` keeps the totals shaped as a column or row instead of flattening them.

```python
# shape [2, 3]
N = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
```

| Code | Result | Shape | What it is |
|---|---|---|---|
| `N.sum(dim=1)` | `[6, 15]` | `[2]` | one total per row |
| `N.sum(dim=1, keepdim=True)` | `[[6], [15]]` | `[2, 1]` | the same, as a column |
| `N.sum(dim=0)` | `[5, 7, 9]` | `[3]` | one total per column |

One way to remember it: `dim=1` is the dimension that **disappears**. The shape `[2, 3]` loses its `3`, so you get one number for each of the 2 rows. `keepdim=True` leaves a `1` in its place, giving `[2, 1]`. You'll see why that matters in the next section.

`mean`, `max` and friends take `dim` the same way.

## Broadcasting {#broadcasting}
> When two tensors have different shapes, PyTorch stretches the smaller one to fit, like a CSS background that repeats to fill a box. Line the shapes up from the right; each pair of sizes must match, or one of them must be 1.

Say you want each row of counts to become probabilities: divide each row by its own total.

```python
P = N / N.sum(dim=1, keepdim=True)
#   [2, 3] / [2, 1]
```

You never wrote a loop. The `[2, 1]` column of totals was **broadcast**, which means stretched sideways, to `[2, 3]`:

```
N        [[1, 2, 3],
          [4, 5, 6]]

totals   [[ 6],          shape [2, 1]
          [15]]

stretch  [[ 6,  6,  6],  now [2, 3]
          [15, 15, 15]]

N / that [[1/6,  2/6,  3/6 ],
          [4/15, 5/15, 6/15]]
```

Every row now adds up to 1. If you've used CSS, think `background-repeat`: a 1-pixel-wide strip repeats until it fills the box. A size of 1 repeats to fill whatever it's lined up against.

**The rule.** Write the two shapes right-aligned. Compare them one position at a time, from the right. Each pair must be equal, or one of the two must be 1, which stretches. A missing size counts as 1.

```
[2, 3]   with  [2, 1]   ✓
 2 = 2, and 1 stretches to 3

[2, 3]   with     [3]   ✓
 missing → 1, stretches to 2;
 3 = 3

[2, 3]   with     [2]   ✗ error
 3 ≠ 2, and neither is 1
```

**The silent bug.** Without `keepdim=True`, the totals have shape `[2]`. Here that's an error, which is good. But makemore's table is **square**, `[27, 27]`. Its row totals without keepdim have shape `[27]`, which lines up with the *columns*, so PyTorch happily divides every **column** by a row total. You get the wrong answer and no error message. Keeping the totals as a `[27, 1]` column is the fix, and it's why the cards keep asking about `keepdim`.

## Whole numbers and decimals {#dtypes}
> Integer tensors can't be averaged. Turn them into floats first with `.float()`.

Counts are whole numbers, so a counts tensor is `int64`. Asking for its average fails:

```python
counts = torch.tensor([1, 2, 4])

counts.mean()
# RuntimeError: mean(): could not
# infer output dtype ...

counts.float().mean()
# tensor(2.3333)
```

Division is the exception. `/` always gives floats, so `N / N.sum(1, keepdim=True)` works directly on counts. When you see a dtype error, check whether you're doing decimal math on whole numbers.

## Setting up Python on your computer {#setup}
> Install Python 3.12, make a virtual environment for the course, and install PyTorch into it. Check that `import torch` works.

The browser runs this lesson's challenges for you, with nothing installed. The last step asks how long your local setup took, and the GPU lessons need Python on your own machine, so set it up now:

1. **Install Python 3.12** from python.org. On Windows, tick "Add python.exe to PATH" in the installer.
2. **Make a project folder and a virtual environment.** A virtual environment is a private set of libraries for one project, a bit like a per-project `node_modules`:
   ```bash
   mkdir zero-to-llm-work
   cd zero-to-llm-work
   python -m venv .venv
   ```
3. **Activate it.** On Windows run `.venv\Scripts\activate`. On macOS or Linux run `source .venv/bin/activate`. Your prompt then starts with `(.venv)`.
4. **Install PyTorch.** On pytorch.org, choose your OS, pip and your CUDA version (if you have an NVIDIA card), then copy the command it gives you. A plain `pip install torch` on Windows installs a CPU-only build. Add NumPy too: `pip install numpy`.
5. **Check it:**
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   You should see a version number and `True` if your GPU is ready, or `False` if you're on CPU. CPU is fine until lesson 0.1.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards and challenges.

- **Tensor**: a grid of numbers with any number of dimensions, one shape and one dtype.
- **Shape**: the size of each dimension, like `[27, 27]`. Read it as "27 rows, 27 columns".
- **Dimension (dim, axis)**: one direction of a tensor. In a `[rows, cols]` grid, dim 0 runs down the rows and dim 1 across the columns.
- **dtype**: the number type of every cell. `int64` is whole numbers and `float32` is decimals.
- **Indexing / slicing**: picking cells (`x[1, 2]`) or ranges (`x[1:3]`). A slice's end is not included.
- **keepdim**: keep a summed dimension as size 1 instead of removing it, so the result still lines up for broadcasting.
- **Broadcasting**: stretching size-1 dimensions so two different shapes can be combined element by element.
- **Comprehension**: a one-line loop that builds a list or dict, like `[stoi[ch] for ch in word]`.
- **Bigram**: a pair of neighbouring characters, like `("e", "m")` in "emma".
- **Vocabulary**: the set of symbols a model knows. For makemore that's 26 letters plus "." for start and end.
- **stoi / itos**: the lookup tables between characters and their numbers ("string to integer" and "integer to string").
- **Virtual environment (venv)**: a private folder of Python libraries for one project.
