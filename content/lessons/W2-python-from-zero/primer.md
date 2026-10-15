---
gist: Python is a way to write step-by-step instructions for a computer. In this lesson you learn five building blocks and use them to write two small functions.
minutes: 15
---

## What this lesson is for {#why}
> You'll learn just enough Python to follow the code in the rest of the course, one small building block at a time.

Every lesson in this course has some code. You don't need to become a programmer first. You need to be able to read short pieces of Python and change them.

This lesson covers five building blocks:

1. **Variables**: giving a value a name.
2. **Lists**: keeping many values in order.
3. **Loops**: doing something for each value.
4. **`if`**: doing something only when it's true.
5. **Functions**: packaging steps so you can reuse them.

That's enough for the next several lessons. Take it slowly. Try the examples in your head before reading the answer.

## Where your code runs {#running}
> In this course, code runs in your browser. Press Run tests and read the first message.

You don't install anything for this lesson. On the **Build** page you'll see:

- **An editor**, where you type code.
- **Run tests**: a button that runs your code and checks it.
- **Tests**: small checks, like "does `count_letter("banana", "a")` give 3?". Each one passes ✓ or fails ✗.

When a test fails, it explains what went wrong in plain words. Read the first message, change one thing, and run again. Getting it wrong a few times is normal. That's how everyone writes code.

## Variables: labelled boxes {#variables}
> A variable is a name for a value. `=` puts a value into it.

```python
age = 30
name = "Ana"
```

Think of each line as putting something into a labelled box:

| Box (variable) | What's inside |
|---|---|
| `age` | `30` |
| `name` | `"Ana"` |

Read `=` as "becomes": "age becomes 30".

You can change what's in a box:

```python
x = 3
x = x + 2
```

The second line works right to left. Python takes what's in `x` (3), adds 2 (that's 5), and puts 5 back into `x`.

To see a value, use `print`:

```python
print(x)
```

This shows `5`.

## Numbers and text {#types}
> Numbers are written plainly. Text goes inside quotes.

```python
count = 4          # a number
word = "banana"    # text
```

Text in quotes is called a **string**. The quotes matter: `"4"` is the text four, and `4` is the number four.

Some things to know:

- `+` adds numbers: `2 + 3` is `5`.
- `/` divides and can give decimals: `7 / 2` is `3.5`.
- `//` divides and throws away the decimals: `7 // 2` is `3`. You'll rarely want this one.
- Lines starting with `#` are **comments**: notes for humans. Python skips them.

## Lists: many values in order {#lists}
> A list keeps values in order. Positions start at 0, not 1.

```python
names = ["ana", "bo", "cy"]
```

Square brackets make a list. Commas separate the items.

Each item has a position, and **counting starts at 0**:

| Position | 0 | 1 | 2 |
|---|---|---|---|
| Item | "ana" | "bo" | "cy" |

So `names[0]` is `"ana"` and `names[2]` is `"cy"`. Starting at 0 feels odd at first. Almost every programming language does it.

Two useful helpers:

- `len(names)` is `3`: how many items.
- `sum([2, 4, 9])` is `15`: everything added up (numbers only).

A string works a lot like a list of letters: `len("banana")` is 6, and `"banana"[0]` is `"b"`.

## Loops: once for each item {#loops}
> `for item in things:` runs the pushed-in lines once for every item.

```python
for name in names:
    print(name)
```

Read it as: "For each item in `names`, call it `name`, and run the lines below."

The line under `for` is pushed in by **4 spaces**. That's how Python knows which lines repeat. Here, `print(name)` runs three times:

| Time round | `name` is | Printed |
|---|---|---|
| 1st | "ana" | ana |
| 2nd | "bo" | bo |
| 3rd | "cy" | cy |

You choose the name after `for`. `for n in names:` works just as well. Pick something that says what one item is.

You can loop over the letters of a word too: `for ch in "abc":` gives `"a"`, then `"b"`, then `"c"`.

## `if`: only when it's true {#if}
> `if` runs its pushed-in lines only when a condition is true. `==` asks "are these equal?".

```python
if ch == "a":
    print("found an a")
```

Watch out for the two kinds of equals sign:

| Write | Meaning |
|---|---|
| `x = 5` | Put 5 into x. |
| `x == 5` | Is x equal to 5? Gives `True` or `False`. |

Mixing these up is the most common beginner mistake. If Python complains about `=` inside an `if`, you probably meant `==`.

## Functions: reusable steps {#functions}
> A function packages steps under a name. `return` hands the answer back.

```python
def double(n):
    result = n * 2
    return result
```

- **`def double(n):`** starts a function called `double`. `n` is the **input**: the value you give it.
- The pushed-in lines are its steps.
- **`return result`** hands the answer back to whoever used the function.

Using it:

```python
print(double(4))
```

Python runs the steps with `n` set to 4, gets 8 back, and prints `8`.

**`return` is not the same as `print`.** `print` only shows something on screen. `return` gives the value back so other code can use it. The tests in this course check what your function *returns*, so a function that only prints will fail them.

## Thinking through the first challenge {#walkthrough}
> Before coding, do the task by hand. Then turn each step you did into a line.

The first challenge: count how many times a letter appears in a word. Try it by hand for "banana" and "a":

| Letter | Is it "a"? | Count so far |
|---|---|---|
| (start) | | 0 |
| b | no | 0 |
| a | yes | 1 |
| n | no | 1 |
| a | yes | 2 |
| n | no | 2 |
| a | yes | 3 |

Look at what you did:

1. You started counting from 0.
2. You looked at each letter in turn. That's a loop.
3. When a letter matched, you added 1. That's an `if`, plus a change to the count.
4. At the end, you said the count. That's `return`.

Each of those is one or two lines of Python. The outline in the editor already has the loop and the `return`. You fill in the rest.

## When Python complains {#errors}
> Error messages look scary but usually point at one small thing.

| You see | It usually means |
|---|---|
| `IndentationError` | The spaces at the start of a line are off. Use exactly 4 per level. |
| `SyntaxError` near `=` | You wrote `=` where you meant `==`, or forgot a `:` after `for`, `if` or `def`. |
| `NameError: name 'x' is not defined` | You used a name before giving it a value, or there's a typo. |
| A test says "gives back nothing" | The function has no `return`, or `return` is in the wrong place. |

The tests in this course also explain mistakes in plain words, so start with the first message in the Tests panel.

## Words you'll see {#glossary}
> The new words from this lesson.

- **Variable**: a name for a value, like a labelled box.
- **String**: text inside quotes, like `"banana"`.
- **List**: values in order inside `[ ]`. Positions start at 0.
- **Loop** (`for`): runs some lines once for each item.
- **Condition** (`if`): runs some lines only when something is true.
- **`==`**: asks "are these equal?". **`=`** stores a value.
- **Function** (`def`): named steps you can reuse.
- **Input** (parameter): the value a function is given, like `n` in `double(n)`.
- **`return`**: hands a function's answer back.
- **Comment** (`#`): a note for people. Python ignores it.
- **Test**: a small check that runs your code and says ✓ or ✗.
