---
id: "0.0"
title: Python bridge and environment
summary: The Python and tensor basics the rest of the course leans on, written as tiny makemore pieces, plus a working local environment.
phase: "0"
hours: 7
type: build
status: ready
protected: false
cut: null
prerequisites: ["W4"]
papers: []
quality_metric:
  id: exercises_passed
  label: Bridge exercises passed
  unit: count
  note: The three browser challenges in this lesson.
  inputs:
    - { id: exercises_passed, label: Challenges passed, hint: Out of 3 }
  compute: value
  reference: 3
  pass: { min: 3 }
systems_metric:
  id: env_setup_minutes
  label: Time to a working environment
  unit: min
  note: From a fresh terminal to `import torch` printing a version on your own machine.
  inputs:
    - { id: env_setup_minutes, label: Minutes to a working environment, hint: "Until `python -c \"import torch\"` runs" }
  compute: value
  reference: 30
  pass: { max: 120 }
quiz:
  - id: mean_shape
    q: "`x` has shape [3, 4] and `m = x.mean(dim=1)` has shape [3]. What does `x - m` do?"
    answer: 1
    options:
      - text: Subtracts each row's mean from that row
        why: "That needs `m` with shape [3, 1], which is what `keepdim=True` gives. A plain [3] lines up with the last dimension, and that one has size 4."
      - text: "Raises an error, because sizes 4 and 3 don't match"
        why: "Broadcasting compares shapes from the right: 4 against 3. They differ and neither is 1, so torch raises \"The size of tensor a (4) must match the size of tensor b (3)\". On a square [4, 4] tensor the same line runs and silently gives the wrong answer. That's step 3."
      - text: Subtracts each column's mean from that column
        why: "Column means come from `dim=0` and would have shape [4]. This `m` holds one mean per row."
  - id: int_mean
    q: "`torch.tensor([1, 2, 3]).mean()` raises an error. Why?"
    answer: 0
    options:
      - text: The tensor holds int64, and mean() needs a float dtype
        why: "Python ints become int64, and the mean of integers usually isn't one. Call `.float()` first: `torch.tensor([1, 2, 3]).float().mean()` gives `tensor(2.)`."
      - text: mean() needs a dim argument
        why: Without `dim`, mean() averages every element. That's fine for a float tensor.
      - text: A 1-D tensor has no mean, so you have to reshape it to 2-D first
        why: Shape isn't the problem. Any float tensor, even a single number, has a mean.
---

## Goal

Get comfortable enough with Python and tensors that the code in the next five lessons reads like prose: lists and dictionaries, comprehensions, indexing, and the one broadcasting bug everyone writes once. Every exercise is a piece of makemore, the character-level name generator you build in 0.3. Alongside it, set up a local Python environment with PyTorch, ready for the GPU check in 0.1.

## Learn

```card
type: concept
id: python_for_programmers
eyebrow: New tools
title: Four Python tools you'll meet in this lesson.
body: |
  - **A dictionary is a lookup table.** `stoi = {"a": 1, "b": 2}` maps letters to numbers, and `stoi["a"]` gives back `1`.
  - **A one-line loop builds a list.** `[stoi[ch] for ch in "ab"]` means "for each letter, look up its number". It gives `[1, 2]`.
  - **A slice copies part of a word.** `"emma"[1:3]` is `"mm"`: it starts at position 1 and stops *just before* 3.
  - **`zip` pairs things up.** `zip("emma", "mma")` pairs each letter with the one after it: e–m, m–m, m–a.

  The primer explains each one slowly, with examples.
```

```card
type: concept
id: tensors_and_broadcasting
eyebrow: Tensors
title: A tensor is an n-dimensional array with one shape and one dtype.
body: |
  `torch.arange(12).reshape(3, 4)` is 3 rows of 4. Python ints become `int64` and floats become `float32`. `print(t.shape, t.dtype)` is the most useful debugging line in this course.

  **Broadcasting** lets a small tensor stretch to fit a big one. Line the shapes up from the *right*. Each pair of sizes must be equal, or one of them must be 1. `[3, 4] - [3, 1]` subtracts one number per row. `[3, 4] - [4]` subtracts one number per column. `[3, 4] - [3]` is an error.
```

```card
type: predict
id: predict_index
eyebrow: Predict the output
title: What prints?
code: |
  x = torch.arange(12).reshape(3, 4)
  print(x[:, 1], x[-1].shape)
options:
  - text: "tensor([1, 5, 9]) torch.Size([4])"
    correct: true
    why: "`x[:, 1]` takes column 1 from every row. `x[-1]` is the last row, and an integer index drops that dimension, so its shape is [4]."
  - text: "tensor([4, 5, 6, 7]) torch.Size([4])"
    why: "That's `x[1]`, row 1. The `:` before the comma means every row, and the 1 picks a column."
  - text: "tensor([1, 5, 9]) torch.Size([1, 4])"
    why: "An integer index removes its dimension. To keep it, slice instead: `x[-1:]` has shape [1, 4]."
  - text: "tensor([[1], [5], [9]]) torch.Size([4])"
    why: "The integer 1 drops the column dimension, so you get a flat [3]. `x[:, 1:2]` would keep it as [3, 1]."
hint: The rows are 0–3, 4–7 and 8–11. A `:` keeps a dimension and an integer removes it.
```

```card
type: fill_blank
id: keepdim_blank
eyebrow: Fill the blanks
title: Turn a table of counts into rows of probabilities.
code: |
  # N is [27, 27]: row = current char, column = next char
  P = N / N.sum(dim=[[0]], keepdim=[[1]])
tokens: ["0", "1", "True", "False"]
answer: ["1", "True"]
tips:
  "0": "`dim=0` adds down each column. You want one total per row, which means adding along it: `dim=1`."
  "False": "Without keepdim the totals have shape [27]. Broadcasting lines a [27] up with the columns, so every entry gets divided by the wrong row's total, with no error. Keep it [27, 1]."
why_ok: "`N.sum(1, keepdim=True)` is a [27, 1] column of row totals, so each row is divided by its own total and sums to 1."
why_wrong: Add along each row, and keep the result as a column so broadcasting pairs every row with its own total.
hint: Which dim runs along a row? And should the totals stay a column, or flatten into a row?
```

```card
type: calc
id: broadcast_count
eyebrow: Do the math
title: How many numbers are in a + b?
given:
  - { label: a.shape, value: "[32, 1, 5]" }
  - { label: b.shape, value: "[4, 5]" }
unit: numbers
answer: { min: 640, max: 640 }
placeholder: "0"
why_ok: "From the right: 5 and 5 match, 1 stretches to 4, and b gets a leading 1 that stretches to 32. The result is [32, 4, 5], which is 640 numbers."
why_wrong: "Line the shapes up from the right and pad b on the left with a 1. Each size of 1 stretches to match the other. Then multiply the result's sizes."
hint: "Pad b to [1, 4, 5], then take the larger size in each position."
```

```card
type: reorder
id: vocab_order
eyebrow: Put it in order
title: Build makemore's vocabulary from a list of names.
lines:
  - chars = sorted(set("".join(words)))
  - "stoi = {ch: i + 1 for i, ch in enumerate(chars)}"
  - stoi["."] = 0
  - "itos = {i: ch for ch, i in stoi.items()}"
shuffle: [2, 0, 3, 1]
why_ok: "The letters come first, then their numbers from 1, then \".\" claims 0, and only then is `itos` built, so it includes \".\" too."
why_wrong: "Follow the names: `stoi` needs `chars`, and `itos` is built from `stoi`, so it has to come after \".\" is added."
hint: "`itos` flips every entry of `stoi`. Which entries should it already contain?"
```

```card
type: quiz
ref: mean_shape
```

```card
type: quiz
ref: int_mean
```

## Build steps

### Encode names as integers

```challenge
id: encode_decode
file: bridge.py
minutes: 20
start: { code: "# Python warm-up: characters to integers and back\n\n" }
setup:
  - say: A language model never sees letters, only integers. First, the lookup tables. "." gets 0 and marks where a name starts and stops; a to z get 1 to 26.
    type: "chars = \"abcdefghijklmnopqrstuvwxyz\"\nstoi = {ch: i + 1 for i, ch in enumerate(chars)}\nstoi[\".\"] = 0\nitos = {i: ch for ch, i in stoi.items()}\n\n"
    pause: 3.2
  - say: "Two dictionaries: stoi maps a character to its integer, so stoi['a'] is 1, and itos goes back, so itos[1] is 'a'."
    pause: 3.0
  - say: "Here's the stub. Your turn: write encode and decode, then run the tests."
    type_file: encode_decode/stub.py
    pause: 1.2
tests: encode_decode/tests.py
solution: encode_decode/solution.py
alternatives:
  - encode_decode/alternatives/concat_zeros.py
  - encode_decode/alternatives/map_get.py
  - encode_decode/alternatives/loops.py
hints:
  start:
    try: |
      word = "emma"
      print("." + word + ".")
      print(stoi["e"])
    see: |
      .emma.
      5
    next: "That's the word with its markers, and one letter turned into its number. `encode` does that lookup for every character of the dotted word. A plain `for` loop that appends to a list is fine."
  logic: |
    encode(word):
      put a "." at the front and at the back of word
      for each character in that longer word:
        look up its integer in stoi
      → return the integers as a list
    decode(ids):
      for each integer in ids:
        look up its character in itos
      glue the characters into one string
      → return it   (keep the "." marks)
  tools: |
    put "." at both ends
      a + b            joins two strings into one
    look up each character's integer (and each integer's character)
      d[key]           the value stored under key in dict d
      [f(x) for x in seq]   builds a list, one entry per item
    glue the characters into one string
      "".join(seq)     glues a sequence of strings together
  nudge: "encode needs a \".\" on each side of the word, then one lookup per character. decode does the lookups the other way and glues the characters together."
  concept: "A list comprehension, `[f(x) for x in seq]`, builds a list in one line. Strings are sequences of characters, and `\".\" + word + \".\"` adds the markers. `\"\".join(seq)` glues a sequence of strings into one."
  example: |
    Trace `encode("ava")`:

    - With the markers: `".ava."`.
    - Characters: `. a v a .`
    - Looked up in `stoi`: `0 1 22 1 0`.
    - So `encode("ava")` is `[0, 1, 22, 1, 0]`.

    Now trace `decode([0, 5, 13, 13, 1, 0])`:

    - Looked up in `itos`: `. e m m a .`
    - Glued together: `".emma."`.

    Check: `decode(encode("emma"))` gives `".emma."`, with the dots still there.
  skeleton: |
    return [stoi[____] for ch in ____]
    return "".____(itos[i] for i in ____)
  solution: |
    return [stoi[ch] for ch in "." + word + "."]
    return "".join(itos[i] for i in ids)
on_solve: "That's the tokenizer for every model in phase 0: 27 symbols, one integer each. Lesson 1.5 replaces it with BPE, which learns its symbols from data."
mistakes:
  - { file: encode_decode/mistakes/placeholders.py, fails: encode_list, says: "`encode` still returns `...`" }
  - { file: encode_decode/mistakes/encode_placeholder.py, fails: encode_list, says: "`encode` still returns `...`" }
  - { file: encode_decode/mistakes/generator.py, fails: encode_list, says: "returns a generator" }
  - { file: encode_decode/mistakes/no_dots.py, fails: encode_values, says: "Where does the model see that a name starts" }
  - { file: encode_decode/mistakes/ord.py, fails: encode_values, says: "Unicode code points from `ord()`" }
  - { file: encode_decode/mistakes/index_zero_based.py, fails: encode_values, says: "the same number as" }
  - { file: encode_decode/mistakes/decode_placeholder.py, fails: decode_string, says: "`decode` still returns `...`" }
  - { file: encode_decode/mistakes/decode_list.py, fails: decode_string, says: "glues a sequence of strings" }
  - { file: encode_decode/mistakes/decode_stoi.py, fails: decode_string, says: "goes from integers to characters?" }
  - { file: encode_decode/mistakes/strips_dots.py, fails: decode_string, says: "drops the 0s" }
```

### Pick probabilities by index

```challenge
id: probs_of
file: tensors.py
minutes: 20
start: { code: "import torch\n\n" }
setup:
  - say: Now tensors. Here's a 3 by 3 table of probabilities. The row is the current character and the column is the next one, so each row sums to 1.
    type: "P = torch.tensor([[0.1, 0.6, 0.3],\n                  [0.5, 0.25, 0.25],\n                  [0.2, 0.2, 0.6]])\n\n"
    pause: 3.0
  - say: "P[0, 1] reads row 0, column 1. And here are four pairs: xs holds the current characters, ys the ones that came next."
    type: "xs = torch.tensor([0, 2, 1, 0])\nys = torch.tensor([1, 0, 2, 2])\n\n"
    pause: 2.6
  - say: "Your turn: write probs_of so it returns the probability of each pair, four numbers in one tensor."
    type_file: probs_of/stub.py
    pause: 1.2
tests: probs_of/tests.py
solution: probs_of/solution.py
alternatives:
  - probs_of/alternatives/stack_loop.py
  - probs_of/alternatives/gather.py
  - probs_of/alternatives/arange_rows.py
hints:
  start:
    try: |
      print(xs[0], ys[0])
      print(P[xs[0], ys[0]])
    see: |
      tensor(0) tensor(1)
      tensor(0.6000)
    next: "That's the first pair: row `xs[0]`, column `ys[0]`, and the one number of `P` it picks. `probs_of` needs that number for all four pairs, in one tensor. Try it for one pair first, then ask what goes inside the brackets to do all four at once."
  logic: |
    pair i means: row xs[i], column ys[i] of P
    read the cell of P for every pair, all at once (no loop)
      → one number per pair
      → return them in one tensor   (4 pairs → 4 numbers)
  tools: |
    read the cell at one row and one column
      t[a, b]          the value in row a, column b of t
    read many cells at once
      t[ra, cb]        ra and cb are index tensors of the same
                       length: one value per matching pair
  nudge: "You could loop over the pairs, but tensors let you index with whole tensors of positions. Can one indexing expression take the row from xs and the column from ys?"
  concept: "Indexing with two tensors of the same length inside one bracket pairs them up: `P[a, b]` gives `P[a[0], b[0]]`, `P[a[1], b[1]]`, and so on. Two separate brackets, `P[a][b]`, index one dimension at a time and do something else."
  example: |
    Use the setup's `P`, `xs = [0, 2, 1, 0]` and `ys = [1, 0, 2, 2]`. The pairs are (0,1), (2,0), (1,2), (0,2):

    - Pair 1, row 0 column 1: `0.6`.
    - Pair 2, row 2 column 0: `0.2`.
    - Pair 3, row 1 column 2: `0.25`.
    - Pair 4, row 0 column 2: `0.3`.

    So `probs_of(P, xs, ys)` is `tensor([0.6, 0.2, 0.25, 0.3])`, shape `[4]`. If you got a `[4, 4]` table or a `[4, 3]` one, you picked rows and columns separately instead of in pairs.
  skeleton: |
    return P[____, ____]
  solution: |
    return P[xs, ys]
on_solve: "That one line is the heart of every loss in this course. In 0.3 it becomes `probs[torch.arange(n), ys]`: the probability the model gave to the character that actually came next."
mistakes:
  - { file: probs_of/mistakes/placeholder.py, fails: returns_tensor, says: "still returns `...`" }
  - { file: probs_of/mistakes/list_loop.py, fails: returns_tensor, says: "returns a Python list of 4 tensors" }
  - { file: probs_of/mistakes/two_brackets.py, fails: one_per_pair, says: "from *every* one of those rows" }
  - { file: probs_of/mistakes/rows_only.py, fails: one_per_pair, says: "one whole row of P per pair" }
  - { file: probs_of/mistakes/swapped.py, fails: right_values, says: "the transpose" }
```

### Normalize rows with broadcasting

```challenge
id: normalize_rows
file: tensors.py
minutes: 25
start: { from: probs_of }
setup:
  - say: "Probabilities come from counts. Here's a 4 by 4 table of how often each character followed each other one. Notice it's square: that matters."
    type_file: normalize_rows/stub.py
    pause: 2.6
  - say: "Your turn: write normalize_rows so every row sums to 1. The last print shows the row sums."
    pause: 1.0
tests: normalize_rows/tests.py
solution: normalize_rows/solution.py
alternatives:
  - normalize_rows/alternatives/dim_kwarg_float.py
  - normalize_rows/alternatives/unsqueeze.py
  - normalize_rows/alternatives/transpose.py
hints:
  start:
    try: |
      print(N[0])
      print(N[0].sum())
      print(N[0] / N[0].sum())
    see: |
      tensor([3, 1, 0, 4])
      tensor(8)
      tensor([0.3750, 0.1250, 0.0000, 0.5000])
    next: "That's row 0 turned into chances that add up to 1. Now every row at once: print `N.sum(1)` and its `.shape`. Each row has to meet its own total, so check which way that shape stretches."
  logic: |
    add up each row of N along the row   (one total per row)
      → keep the totals as a column, not a flat list
      → divide every entry by the total of its own row
      → return the table   (fractions, so each row sums to 1)
  tools: |
    add up each row
      t.sum(dim)       adds up t along dimension dim
      (sum has an option to keep the summed dimension as size 1)
    divide every entry by its row's total
      a / b            divides entry by entry; shapes line up
                       from the right (broadcasting)
  nudge: Each row needs its own total. Which dim adds along a row, and what shape does the result need so each row meets its own total?
  concept: "`N.sum(1)` gives one total per row, shape [4]. Broadcasting lines shapes up from the right, so a [4] stretches across the *columns*, the wrong way, and on a square table nothing complains. `keepdim=True` keeps the totals as a [4, 1] column, which stretches across each row. `/` on integers gives floats, so the counts can stay int64."
  example: |
    Trace it on a small table, `N = [[1, 3], [1, 7]]`:

    - Row totals: `[4, 8]`. Kept as a column, the shape is `[2, 1]`.
    - Row 0 is divided by `4`: `[0.25, 0.75]`.
    - Row 1 is divided by `8`: `[0.125, 0.875]`.
    - Row sums: `[1.0, 1.0]`.

    Now the bug. If the totals stay flat, shape `[2]`, the `4` and the `8` get matched to the columns instead: `[[0.25, 0.375], [0.25, 0.875]]`. No error appears, but the row sums come out `[0.625, 1.125]`, not 1.
  skeleton: |
    return N / N.____(___, keepdim=____)
  solution: |
    return N / N.sum(1, keepdim=True)
on_solve: "Now delete `keepdim=True` and run it again. Nothing fails, but the row sums go wrong. Karpathy hits this exact bug in the makemore video, and you'll see it diagnosed again in 0.3."
mistakes:
  - { file: normalize_rows/mistakes/placeholder.py, fails: returns_floats, says: "still returns `...`" }
  - { file: normalize_rows/mistakes/floor_div.py, fails: returns_floats, says: "`//` is floor division" }
  - { file: normalize_rows/mistakes/no_keepdim.py, fails: rows_sum_to_one, says: "even though no error appeared" }
  - { file: normalize_rows/mistakes/columns.py, fails: rows_sum_to_one, says: "Your *columns* sum to 1" }
  - { file: normalize_rows/mistakes/grand_total.py, fails: rows_sum_to_one, says: "divided by the grand total" }
  - { file: normalize_rows/mistakes/inverted.py, fails: rows_sum_to_one, says: "the right way up?" }
```

### Record your metrics

```metrics
id: record
intro: How many of the three challenges passed, and how long your local setup took, from a fresh terminal to `import torch` working.
```

## Hints

- Use Python 3.11 or 3.12 and a virtual environment per project: `python -m venv .venv`, activate it, then `pip install torch numpy`. For an NVIDIA GPU, copy the install command for your CUDA version from the selector on pytorch.org; the plain `pip install torch` on Windows gives you a CPU-only build.
- Check the install with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`. Lesson 0.1 goes further and times a matmul on the GPU.
- The browser runs a small NumPy copy of torch (mini-torch). Everything in these challenges runs the same on your machine with real PyTorch.
- When a shape surprises you, print it. `print(t.shape, t.dtype)` settles most arguments with broadcasting.

## Done when

- [ ] encode and decode round-trip any name {#encode_decode auto=challenge:encode_decode}
- [ ] Rows normalize without the keepdim bug {#normalize_rows auto=challenge:normalize_rows}
- [ ] On your machine, `python -c "import torch, numpy; print(torch.__version__)"` prints a version inside a virtual environment {#local_env}
- [ ] Exercises passed and setup time recorded {#metrics_recorded auto=metrics}
