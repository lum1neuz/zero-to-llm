---
gist: Training means turning each of a model's knobs a little in whichever direction lowers the loss. This lesson shows how to work out that direction, first by hand and then with a tiny program you write yourself.
minutes: 25
---

## Start here {#why}
> To train a model, you need to know for every knob: if I turn it up a little, does the loss go up or down, and by how much? This lesson works that out.

In W1 you met the idea that a model has many adjustable numbers inside it, like knobs on a mixing desk. In 0.1 you met the **loss**: one number that says how bad the model's guesses are. Lower is better.

Training repeats four steps:

1. Run the model and work out the loss.
2. For every knob, find out how the loss would change if you turned that knob up a tiny bit.
3. Turn every knob a little in the direction that makes the loss smaller.
4. Go back to step 1.

Step 2 is the clever part. A real model has millions of knobs. PyTorch does step 2 for you with one line, `loss.backward()`. The part of PyTorch that does it is called **autograd**.

In this lesson you build a tiny autograd yourself. It's called **micrograd**, and it was designed by Andrej Karpathy. It works on single numbers instead of grids, and it's about 100 lines long. Once you've built it, `loss.backward()` in the next lessons won't be a mystery.

## What you need from before {#before}
> You need Python functions and variables, and the idea from 0.1 that the loss is one number where lower is better.

| Idea | Where you met it |
|---|---|
| Variables, `def` functions, `return` | W2 |
| `x += 1` adds 1 to x | W2 |
| Powers, like 3² = 9 | W4 |
| The loss: one number, lower is better | 0.1 |
| `print(a, b)` | W2 |

Everything else, including the maths, is explained as it comes up.

## Nudge it and watch {#nudge}
> To see how much an output depends on an input, nudge the input by a tiny amount and see how much the output moves.

Think of a recipe. If you add one more spoon of sugar, how much sweeter does the cake get? You could find out by trying it.

We'll do the same with numbers. Here is a small rule, written as a function:

```python
def f(x):
    return 3 * x**2 - 4 * x + 5
```

In words: square x, multiply by 3, take away 4 times x, then add 5. (`x**2` means x squared.)

Let's see what happens near x = 3.

| x | Working | f(x) |
|---|---|---|
| 3 | 3·9 − 12 + 5 | 20 |
| 3.0001 | 3·9.0006 − 12.0004 + 5 | about 20.0014 |

We nudged x up by 0.0001, and f went up by about 0.0014. That's 14 times as much as the nudge.

So near x = 3, **f changes 14 times as fast as x**. This number, 14, is called the **slope**, or the **derivative** of f at x = 3.

- A **positive** slope means: turn x up, and f goes up.
- A **negative** slope means: turn x up, and f goes down.
- A **bigger** slope (ignoring the sign) means f reacts more strongly.

In symbols, people write the slope of f as **df/dx**. Read it as "how much f changes for each bit x changes".

## Shortcuts for slopes {#rules}
> You don't have to nudge every time. A few simple rules give the slope directly, and nudging is how you check them.

Nudging works, but it's slow and a little inexact. For common rules there are exact shortcuts:

| Rule | In words | Slope |
|---|---|---|
| a fixed number, like 5 | doesn't change when x changes | 0 |
| x | changes exactly as x does | 1 |
| x² | bring the 2 down in front, then lower the power by one | 2x |
| any power, xᵏ | bring k down in front, lower the power by one | k · xᵏ⁻¹ |
| eˣ | e is a special number, about 2.718 | eˣ (its own slope) |

Check the power rule against our nudge. In f, the 3x² part has slope 3 · 2x = 6x, the −4x part has slope −4, and the 5 has slope 0. Added together: 6x − 4. At x = 3 that's 18 − 4 = **14**. It matches the nudge.

**tanh** is one more function you'll need. It squashes any number into the range −1 to 1:

| x | −3 | −1 | 0 | 0.5 | 1 | 3 |
|---|---|---|---|---|---|---|
| tanh(x) | −0.995 | −0.762 | 0 | 0.462 | 0.762 | 0.995 |

Near 0 it changes quickly. Far from 0 it flattens out near −1 or 1. Its slope rule is **1 − tanh(x)²**: one, minus the output squared. At x = 0 that's 1 − 0 = 1. At x = 3 it's 1 − 0.995² ≈ 0.01, almost flat. Notice the rule uses tanh's *output*, not x itself.

## Two inputs: one slope each {#two-inputs}
> When a result depends on two numbers, nudge them one at a time. For a times b, the slope for a is b, and the slope for b is a.

Let a = 2 and b = −3. Their product is c = a × b = −6.

Nudge a, keep b fixed:

| | a | b | c |
|---|---|---|---|
| before | 2 | −3 | −6 |
| after | 2.001 | −3 | −6.003 |

c moved by −0.003 when a moved by 0.001. That's −3 times the nudge. And −3 is b.

Now nudge b, keep a fixed:

| | a | b | c |
|---|---|---|---|
| before | 2 | −3 | −6 |
| after | 2 | −2.999 | −5.998 |

c moved by +0.002. That's 2 times the nudge. And 2 is a.

So for a product, **each input's slope is the other input's value**. For a sum, c = a + b, nudging either input moves c by exactly the nudge, so both slopes are 1.

These are called **local slopes** (or local derivatives): how one step reacts to each of its own inputs. A very common mistake is to give each input its own value instead of the other one's. The first challenge checks for it.

## A chain of steps: multiply the slopes {#chain}
> When one thing feeds the next, the slopes multiply along the chain. This is called the chain rule.

Think of changing money with made-up rates. One dollar buys 0.9 euros, and one euro buys 160 yen. How many yen does one dollar buy? You multiply along the chain: 0.9 × 160 = 144.

Slopes work the same way. Carry on the example: c = a × b, and then L = 2 × c.

- L changes **2** times as fast as c.
- c changes **−3** times as fast as a (that's b).
- So L changes 2 × (−3) = **−6** times as fast as a.

Check it with a nudge. Set a = 2.001. Then c = −6.003 and L = −12.006. Before, L was −12. So L moved by −0.006 when a moved by 0.001. That's −6 times the nudge.

This is the **chain rule**: to find how the end of a chain reacts to the start, multiply the local slopes along the way.

In micrograd, each number has a field called `grad`. It holds the slope of the final output with respect to that number. "grad" is short for **gradient**, the usual word for these slopes.

## Python classes, gently {#classes}
> A class is a blueprint for making objects that carry their own data. micrograd makes one object for every number in a calculation.

A **class** is a blueprint. Each object made from it carries its own data. Here's a small one:

```python
class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0.0
```

| Line | What it means |
|---|---|
| `class Value:` | start a blueprint called Value |
| `def __init__(self, data):` | runs every time you make a new Value |
| `self.data = data` | store the number in this object |
| `self.grad = 0.0` | every Value starts with a gradient of 0 |

`self` means "this particular object". You make one like this:

```python
a = Value(2.0)
print(a.data)
print(a.grad)
```

That prints `2.0` and `0.0`.

**Teaching `+` new tricks.** If a class has a method named `__add__`, Python runs it whenever you write `a + b`. In the same way, `__mul__` runs for `*`, and `__pow__` runs for `**`. That's how micrograd makes `a * b` build a new Value.

**A function inside a function.** A function written inside another one can use the outer function's variables, even later:

```python
def make_greeter(name):
    def greet():
        print("hi", name)
    return greet
```

`make_greeter("emma")` returns a function. Calling it later prints `hi emma`: it remembered `name`. micrograd uses this so each result remembers which Values made it.

(Side note for Java readers: `self` is `this`, `__init__` is the constructor, and the inner function is like a lambda that captures variables.)

## The Value and its graph {#graph}
> Every Value remembers which Values it was made from. Together they form a map of the whole calculation, from inputs to output.

The challenges use one small **neuron**: a few inputs times weights, plus a number called a **bias**, put through tanh. Here it is, o = tanh(x × w + b), with the lesson's numbers:

```
x = 2.0     w = −0.5
     \       /
      ( * )
        │
   xw = −1.0     b = 0.75
          \       /
           ( + )
             │
        n = −0.25
             │
          (tanh)
             │
       o = −0.2449
```

Each box is a Value. Each Value has five fields:

| Field | Holds |
|---|---|
| `data` | the number itself, like −0.25 |
| `grad` | the slope of the output with respect to this Value. Starts at 0. |
| `_prev` | the Values it was made from |
| `_op` | which step made it, like `"*"` |
| `_backward` | a small function that passes this Value's grad back to the Values it came from |

Working left to right, from inputs to output, is called the **forward pass**. Filling in every `grad`, from the output back to the inputs, is the **backward pass**, or **backpropagation**.

## Walking backwards by hand {#by-hand}
> Start with the output's grad set to 1. Then each step multiplies the grad it received by its local slopes and passes the result back to its inputs.

The output's slope with respect to itself is 1: if you nudge o, o moves by exactly the nudge. So o.grad = 1.

Now walk back, one step at a time:

| Step | Local slope | Grad passed back |
|---|---|---|
| tanh (o from n) | 1 − o² = 1 − 0.0600 = 0.9400 | n.grad = 0.94 × 1 = 0.94 |
| + (n from xw and b) | 1 for each input | xw.grad = 0.94, b.grad = 0.94 |
| * (xw from x and w) | for x it's w; for w it's x | x.grad = −0.5 × 0.94 = −0.47, w.grad = 2 × 0.94 = 1.88 |

Read x.grad = −0.47 like this: nudge x up a little, and o goes *down*, at about half the rate. That's because w is negative.

Two mistakes the tests look for:

- **tanh with the wrong number.** The slope is 1 minus the *output* squared (o = −0.2449), not the input n.
- **`.grad` instead of `.data`.** For a product, the local slope is the other input's *value*. Its grad is a different number, and it's still 0 at that moment.

## Adding up, and doing it in order {#order}
> If a Value is used twice, it gets gradient from both uses, so grads are added with `+=`. And each step must wait until everything after it has sent its grad back.

**Adding up.** Take b = a + a, with a = 3. Nudge a by 0.001 and b moves by 0.002, so the slope is 2. Each use of a sends back 1, and only adding them gives 2. If you used `=` instead of `+=`, the second use would overwrite the first, and you'd get 1.

This happens all the time in real models. The same knob is used for many examples, and its slopes all add up.

It also means grads keep adding up between training steps, so a training loop sets every grad back to 0 before each backward pass.

**Doing it in order.** A Value's grad is only complete once every step that used it has sent its share back. Think of getting dressed: socks before shoes. Getting undressed goes the other way: shoes off before socks.

The backward pass is the "undressing" order. First you list every Value so that each one comes after the Values it was made from. That list is called a **topological order**. Then you walk it backwards, from the output to the inputs.

The step-2 challenge asks you to build that list. Two details matter, and the tests check both:

- **Remember who you've already listed.** A Value can be reached along more than one path. Without a "visited" check, it gets listed twice, and its grad is passed back twice.
- **List a Value after the Values it came from, not before.** Otherwise the backward walk reaches it too early.

## More steps, and checking against PyTorch {#more-ops}
> New steps only need two things: what they compute, and their local slope. Then you check your grads against PyTorch's.

Step 3 adds two more steps: powers (`x ** k`) and `exp` (e to the power x). Division is then built from them, because a divided by b is a times b to the power −1. Their local slopes come from the shortcuts table. A common slip with powers is to lower the power but forget to bring the old power down in front, or the other way round.

With those, you can rebuild tanh from smaller pieces and check that the grads come out the same:

```
tanh(x) = (e^2x − 1) / (e^2x + 1)
```

To check your engine, the challenges do the same calculation in PyTorch:

```python
x = torch.tensor(
    2.0,
    dtype=torch.float64,
    requires_grad=True)
```

| Part | Meaning |
|---|---|
| `2.0` | the number |
| `dtype=torch.float64` | store it with 64 bits, the same precision as a plain Python number |
| `requires_grad=True` | ask PyTorch to track this number, like a micrograd Value |

After `o.backward()`, PyTorch's `x.grad` holds the slope, just like yours. The step prints the biggest difference between your grads and PyTorch's. That's this lesson's quality score: anything under 0.000001 passes, and a correct engine prints 0.0. The other score counts the steps that have their own `_backward`: +, *, tanh, ** and exp make 5.

**What the grads are for.** Once every knob has its grad, training turns each knob a small step against it. A positive grad means "turning this up makes the loss worse", so you turn it down. The size of the step is set by the **learning rate**, for example 0.05. Too small and training crawls; too big and it overshoots. Lesson 0.3 runs this loop for real.

## Words you'll see {#glossary}
> A short list of the new words in this lesson's cards, challenges and scores.

- **Parameter (weight, knob)**: an adjustable number inside a model.
- **Slope (derivative)**: how much an output changes for each bit an input changes, for a tiny nudge.
- **Local slope (local derivative)**: how one step reacts to one of its own inputs.
- **Gradient (grad)**: the slope of the final output, usually the loss, with respect to one number.
- **Chain rule**: along a chain of steps, the slopes multiply.
- **tanh**: a function that squashes any number into −1…1. Its slope is 1 − tanh².
- **Neuron**: inputs times weights, plus a bias, through tanh.
- **Bias**: the number added in a neuron, b.
- **Class / object**: a blueprint, and one thing made from it. `self` means "this object".
- **Value**: micrograd's number that remembers how it was made.
- **Graph**: the map of Values and the steps that connect them.
- **Forward pass**: working out the output from the inputs.
- **Backward pass (backpropagation)**: working out every grad, from the output back to the inputs.
- **Topological order**: a list where everything comes after what it was made from.
- **Autograd**: software that does the backward pass for you, like micrograd or PyTorch.
- **`requires_grad`**: PyTorch's switch to track a number for autograd.
- **float64**: numbers stored with 64 bits, the same precision as plain Python numbers.
- **Learning rate**: how big a step training takes against the gradient.
- **Max diff**: the biggest gap between your grads and PyTorch's. This lesson's quality score.
