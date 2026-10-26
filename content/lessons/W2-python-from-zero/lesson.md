---
id: "W2"
title: Python from zero
summary: Write your first lines of Python, in the browser, nothing to install. Variables, lists, loops and functions, one at a time.
phase: "W"
hours: 2
type: intro
status: ready
protected: false
cut: null
prerequisites: ["W1"]
papers: []
quality_metric:
  id: challenges_solved
  label: Challenges solved
  unit: count
systems_metric:
  id: minutes_spent
  label: Time spent on this lesson
  unit: min
quiz:
  - id: equals_sign
    q: In Python, what does `x = 5` do?
    answer: 0
    options:
      - text: Puts the value 5 into a box called x
        why: "Right. `=` stores a value. Read it as \"x becomes 5\". To *ask* whether two things are equal, Python uses `==`."
      - text: Checks whether x is equal to 5
        why: "That's `==`, with two equals signs. A single `=` stores a value."
      - text: Prints the number 5
        why: "Nothing is printed. To show a value on screen you use `print(x)`."
  - id: why_indent
    q: Why are some lines pushed to the right with spaces?
    answer: 1
    options:
      - text: Only to make the code look tidy
        why: "In many languages that's true, but in Python the spaces matter. They change what the code does."
      - text: To show which lines belong inside a loop, an if, or a function
        why: "Yes. Python uses the spaces (4 per level) to group lines. Lines pushed in under `for` repeat; lines back at the left run once afterwards."
      - text: Python ignores spaces, so it makes no difference
        why: "Python cares about spaces at the start of a line. Moving a line in or out changes when it runs."
---

## Goal

Read and write simple Python: store values in variables, keep many values in a list, repeat with a loop, decide with `if`, and package steps into a function. You'll write two tiny functions and check them with real tests, right in your browser.

## Learn

```card
type: concept
id: instructions
eyebrow: What code is
title: A program is a list of instructions, run from top to bottom.
body: |
  Python reads your lines one at a time, from the top.

  ```python
  name = "Ana"
  print("Hello", name)
  ```

  - **`name = "Ana"`** puts the text `"Ana"` into a box called `name`. That box is a **variable**.
  - **`print(...)`** shows something on screen. This line shows **Hello Ana**.

  Text goes in quotes. Numbers don't: `age = 30`.
```

```card
type: predict
id: change_a_variable
eyebrow: Predict the output
title: What gets printed?
code: |
  x = 3
  x = x + 2
  print(x)
options:
  - text: "5"
    correct: true
    why: "Line 1 puts 3 in x. Line 2 takes what's in x (3), adds 2, and puts the result (5) back in x. Then 5 is printed."
  - text: "3"
    why: "Line 2 changes x. `x = x + 2` means \"the new x is the old x plus 2\"."
  - text: "x + 2"
    why: "Python works out `x + 2` as a number before storing it. print shows the number, not the formula."
hint: Read line 2 from the right side first. Work out x + 2, then store it in x.
```

```card
type: concept
id: lists_and_loops
eyebrow: Many values
title: A list holds many values. A loop visits each one.
body: |
  ```python
  names = ["ana", "bo", "cy"]
  for name in names:
      print(name)
  ```

  - **`[...]`** makes a **list**: values in order, separated by commas.
  - **`for name in names:`** means "for each value in the list, call it `name`, and run the lines below".
  - The line pushed in by **4 spaces** is the part that repeats. It runs 3 times, once for each name.

  The output is `ana`, then `bo`, then `cy`, each on its own line.
```

```card
type: fill_blank
id: loop_blanks
eyebrow: Fill the blanks
title: Print every word in the list.
code: |
  words = ["hi", "hey", "yo"]
  for [[0]] in words:
      print([[1]])
tokens: [word, words, word, "print"]
answer: [word, word]
tips:
  words: "`words` is the whole list. After `for`, you choose a new name for *one* item at a time, like `word`."
  print: "`print` is the instruction itself. Inside the brackets goes the thing to show: the current item."
why_ok: "`for word in words:` takes one item at a time and calls it `word`. Then `print(word)` shows that item. It runs 3 times."
why_wrong: "After `for` comes a new name for one item. Then print that same name."
hint: You need a name for one item at a time, and then you print that name.
```

```card
type: reorder
id: function_order
eyebrow: Put it in order
title: Build a function that doubles a number, then use it.
lines:
  - "def double(n):"
  - "    result = n * 2"
  - "    return result"
  - print(double(4))
shuffle: [3, 1, 0, 2]
why_ok: "`def` starts the function. The pushed-in lines are its steps. `return` hands the answer back, and the last line uses the function. It prints 8."
why_wrong: "A function has to be defined (`def`) before you can use it, and `return` comes after the result is worked out."
hint: "The line starting with `def` names the function. Everything pushed in belongs to it."
```

```card
type: calc
id: list_length
eyebrow: Do the math
title: "What does print(len([7, 1, 4, 4])) show?"
given:
  - { label: "len(...)", value: how many items are in a list }
unit: items
answer: { min: 4, max: 4 }
placeholder: "0"
why_ok: "There are four items: 7, 1, 4 and 4. `len` counts them all, even the repeated 4."
why_wrong: "`len` counts items, not their values. Count the numbers between the commas."
hint: Count the values, including repeats.
```

```card
type: quiz
ref: equals_sign
```

```card
type: quiz
ref: why_indent
```

## Build steps

### Count a letter in a word

```challenge
id: count_letter
file: letters.py
minutes: 20
start: { code: "" }
setup:
  - say: "Your first function. It takes a word and a letter, and counts how many times the letter appears. \"banana\" has three a's."
    pause: 3
  - say: "Here's the outline. Your turn: start the counter, then add 1 each time the letter matches. Press Run tests when you're ready."
    type_file: count_letter/stub.py
    pause: 1
tests: count_letter/tests.py
solution: count_letter/solution.py
alternatives:
  - count_letter/alternatives/plus_equals.py
  - count_letter/alternatives/builtin_count.py
hints:
  start:
    try: |
      word = "banana"
      print(word[0], word[0] == "a")
      print(word[1], word[1] == "a")
    see: |
      b False
      a True
    next: "That's the first two letters of `banana`, each checked against `\"a\"`: `False` for `b`, `True` for `a`. A `for` loop gives you each letter in turn, so you can do this check for every one. Make a counter before the loop, and make it bigger only when the check is `True`."
  logic: |
    count starts at 0   (nothing counted yet)
    for each letter ch in word:
      ch is the same as letter?
        yes → count goes up by 1
        no  → leave count alone
    after the loop is finished
      → return count
  tools: |
    start a counter
      name = value     plain assignment: store a value in a name
    go through the word letter by letter
      for ch in seq:   ch is each item of seq in turn
    is ch the same as letter?
      a == b           True if a and b are equal
    add 1 and keep it
      plain arithmetic: a + 1, then store it back in the name
    hand the answer back
      return x         gives x back to whoever called
  nudge: Before you look at any letters, how many matches have you counted?
  concept: "Start `count` at 0. The loop gives you one letter at a time as `ch`. When `ch` is the same as `letter` (use `==` to compare), store a bigger count: `count = count + 1`."
  example: |
    Trace it on `word = "banana"` and `letter = "a"`:

    - Start: `count` is `0`.
    - `b`: not an `a`, so `count` stays `0`.
    - `a`: a match, so `count` becomes `1`.
    - `n`: no match, so `count` stays `1`.
    - `a`: a match, so `count` becomes `2`.
    - `n`: no match, so `count` stays `2`.
    - `a`: a match, so `count` becomes `3`.

    The loop ends, and the function returns `3`. If you print `count` inside your loop, you should see it go `0, 1, 1, 2, 2, 3`.
  skeleton: |
    count = ___
    for ch in word:
        if ch ___ letter:
            count = count + ___
  solution: |
    count = 0
    if ch == letter:
        count = count + 1
on_solve: "Solved. You wrote a loop, an if and a function. That's most of the Python you need for the next lessons."
mistakes:
  - { file: count_letter/mistakes/placeholders.py, fails: returns_number, says: "is still `...`" }
  - { file: count_letter/mistakes/no_return.py, fails: returns_number, says: "gives back nothing" }
  - { file: count_letter/mistakes/never_adds.py, fails: counts_matches, says: "never goes up" }
  - { file: count_letter/mistakes/return_in_loop.py, fails: counts_matches, says: "indented inside the loop" }
  - { file: count_letter/mistakes/counts_everything.py, fails: counts_matches, says: "every letter in the word" }
```

### Average a list of numbers

```challenge
id: average
file: average.py
minutes: 15
start: { code: "" }
setup:
  - say: "An average is the total divided by how many numbers there are. For 2, 4 and 9 the total is 15, and there are 3 numbers, so the average is 5."
    pause: 3.5
  - say: "Your turn: fill in the total, then return the average. Python's sum() adds up a list, and len() counts it."
    type_file: average/stub.py
    pause: 1
tests: average/tests.py
solution: average/solution.py
alternatives:
  - average/alternatives/loop_sum.py
hints:
  start:
    try: |
      numbers = [2, 4, 9]
      print(sum(numbers))
      print(len(numbers))
    see: |
      15
      3
    next: "That's the two numbers an average needs: the total, and how many numbers there are. Put the first one in `total`. Then think about how to combine the two to get the average, `5`."
  logic: |
    add up all the numbers   → total
    count how many numbers there are
      → average = total ÷ that count   (keep the decimals)
      → return the average
  tools: |
    add up all the numbers
      sum(seq)         adds every number in a list
    count the numbers
      len(seq)         how many items are in a list
    divide
      a / b            plain division, keeps the decimals
  nudge: Which two numbers do you need for an average?
  concept: "`sum(numbers)` adds up the whole list. `len(numbers)` tells you how many numbers it has. Divide the first by the second with a single `/`."
  example: |
    Trace it on `numbers = [2, 4, 9]`:

    - The total is 2 + 4 + 9 = `15`.
    - There are `3` numbers in the list.
    - `15 / 3` is `5.0`, the average.

    Another one, `numbers = [1, 2]`: the total is `3`, there are `2` numbers, and `3 / 2` is `1.5`. A single `/` keeps the `.5`.
  skeleton: |
    total = ___(numbers)
    return total / ___(numbers)
  solution: |
    total = sum(numbers)
    return total / len(numbers)
on_solve: "Solved. Averages are everywhere in machine learning. The \"loss\" you'll meet in W4 and 0.3 is an average too."
mistakes:
  - { file: average/mistakes/placeholders.py, fails: small_lists, says: "placeholder" }
  - { file: average/mistakes/returns_total.py, fails: small_lists, says: "which is the total" }
  - { file: average/mistakes/divides_by_three.py, fails: small_lists, says: "always divide by 3" }
  - { file: average/mistakes/floor_division.py, fails: small_lists, says: "throws away everything after the decimal point" }
  - { file: average/mistakes/divides_by_max.py, fails: small_lists, says: "Add the numbers up, then divide" }
```

## Hints

- Nothing to install: the code runs in your browser. Press **Run tests** and read the first message. It tells you what to change.
- If you see an error about spaces or "indent", check that lines inside `for`, `if` and `def` are pushed in by exactly 4 spaces.
- Stuck? Open the hints one at a time. Each one costs a little XP, which is fine.

## Done when

- [ ] Finish the Learn cards {#cards auto=learn}
- [ ] Count a letter in a word {#count auto=challenge:count_letter}
- [ ] Average a list of numbers {#avg auto=challenge:average}
