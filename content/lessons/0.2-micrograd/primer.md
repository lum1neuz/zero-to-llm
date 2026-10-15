---
gist: Training means nudging every number in a model in the direction that lowers the loss. Autograd finds that direction for you; in this lesson you build a small one by hand, one number at a time, and check it against PyTorch.
minutes: 20
---

## What this lesson is for {#why}
> To improve a model you need to know, for every adjustable number, which way to move it and how much. That's the gradient, and autograd computes it.

In 0.1 you met the **loss**: one number that says how bad a model's predictions are. Training is a loop that makes the loss smaller:

1. Run the model and compute the loss.
2. For every adjustable number in the model, find out how the loss would change if that number moved a little.
3. Move every number a small step in the direction that lowers the loss.
4. Repeat, thousands of times.

Step 2 is the hard part. A real model has millions of adjustable numbers, called **parameters** or **weights**. PyTorch does step 2 automatically with one call, `loss.backward()`. That machinery is called **autograd**.

This lesson builds a tiny autograd engine called **micrograd** (it's Andrej Karpathy's design) that works on single numbers instead of tensors. It's about 100 lines. Once you've written it, `loss.backward()` in 0.3 and every later lesson stops being magic.

## What you need from before {#before}
> From 0.0: Python functions, `+=`, and printing. From 0.1: the loss is one number, and lower is better.

- Python has no braces: indentation marks a block, and `def` defines a function.
- `x += 1` adds to `x` in place.
- `print(a, b)` prints both values.
- The loss is a single number computed from the model's predictions. Lower is better.

The new Python in this lesson is classes, operator overloading and closures. The next section covers them.

## Python classes for a Java reader {#classes}
> A Python class is a Java class with `self` instead of `this`, `__init__` instead of a constructor, and special method names that let `a + b` call your own code.

```python
class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0.0

    def __add__(self, other):
        return Value(
            self.data + other.data)

a = Value(2.0)
b = Value(3.0)
c = a + b    # calls a.__add__(b)
c.data       # 5.0
```

| Java | Python |
|---|---|
| `this.data` | `self.data` |
| constructor `Value(double d)` | `def __init__(self, data)` |
| fields declared up front | fields created by assigning `self.x = …` |
| no operator overloading | `__add__` runs for `+`, `__mul__` for `*`, `__pow__` for `**` |
| `toString()` | `__repr__` |

Every method takes `self` as its first parameter explicitly; you don't pass it when you call the method. Method names that start with an underscore, like `_backward`, are "private by convention". Python doesn't enforce it.

**Closures.** A function defined inside another function can use the outer function's variables, like a Java lambda that captures local variables:

```python
def make_greeter(name):
    def greet():
        print("hi", name)
    return greet

g = make_greeter("emma")
g()   # prints: hi emma
```

micrograd uses this to give every result a small function, `_backward`, that remembers which numbers produced it.

## A derivative is a slope {#derivative}
> The derivative of f at x says how much f changes for a tiny nudge of x: nudge x by h, and f moves by about slope × h.

Take `f(x) = 3x² − 4x + 5` and ask how it changes near x = 3. Nudge x by a tiny h = 0.0001 and compare:

```
f(3)      = 27 − 12 + 5 = 20
f(3.0001) ≈ 20.0014

change ≈ 0.0014
slope  ≈ 0.0014 / 0.0001 = 14
```

That ratio, change in output over change in input as the nudge gets tiny, is the **derivative**. Positive means "increasing x increases f"; negative means the opposite; the size says how strongly.

You can also get it exactly from rules, without nudging. The ones this lesson uses:

| Function of x | Derivative |
|---|---|
| a constant, like 5 | 0 |
| x | 1 |
| x², x³, any power xᵏ | k · xᵏ⁻¹ (bring the power down, lower it by one) |
| eˣ | eˣ (it's its own slope) |
| tanh(x) | 1 − tanh(x)² |

For f = 3x² − 4x + 5 the rules give 6x − 4, which is 14 at x = 3, matching the nudge. Each part of a sum gets its own derivative, and the results add.

Nudging is how you *check* a derivative. The rules are how micrograd *computes* it.

## Several inputs: one slope each {#partial}
> When a result depends on several numbers, nudge one at a time. For c = a · b, nudging a moves c by b per unit, and nudging b moves it by a.

Let a = 2 and b = −3, so c = a · b = −6.

```
nudge a: 2.001 · −3 = −6.003
  c moved −0.003 for +0.001
  slope wrt a = −3  (that's b)

nudge b: 2 · −2.999 = −5.998
  c moved +0.002 for +0.001
  slope wrt b = 2   (that's a)
```

So for a product, each input's slope is the **other** input's value. For a sum, c = a + b, nudging either input moves c by exactly the nudge, so both slopes are 1.

These per-input slopes are called **local derivatives**: how one operation's output responds to each of its own inputs, ignoring everything else. A common slip is to swap them, giving a its own value instead of b's. The predict card and the first test check exactly that.

## The chain rule {#chain}
> If L depends on c and c depends on a, the slope of L with respect to a is the two slopes multiplied: dL/da = dL/dc × dc/da.

Continue the example: c = a · b, and then L = 2 · c. How does L respond to a?

- L responds to c with slope 2.
- c responds to a with slope b = −3.
- So L responds to a with slope 2 × (−3) = **−6**.

Check by nudging: a = 2.001 gives c = −6.003 and L = −12.006, which is −0.006 for a nudge of 0.001, a slope of −6.

Think of gears: if the first gear turns the second twice as fast, and the second turns the third three times as fast, the first turns the third six times as fast. The chain rule multiplies slopes along the path from the input to the loss.

The notation dL/da just means "the slope of L with respect to a". In micrograd, every Value's `grad` field holds exactly that: the slope of the final output with respect to this Value.

## The Value graph {#graph}
> Every Value remembers the Values it was made from and the operation that made it. Together they form a graph from the inputs to the output.

Here is the neuron the challenges use, o = tanh(x · w + b), with the lesson's numbers:

```
x = 2.0     w = −0.5
            /
      ( * )
        │
   xw = −1.0     b = 0.75
                 /
           ( + )
             │
        n = −0.25
             │
          (tanh)
             │
       o = −0.2449
```

Each box is a Value with five fields:

| Field | Holds |
|---|---|
| `data` | the number itself, like −0.25 |
| `grad` | slope of the output with respect to this Value; starts at 0 |
| `_prev` | the Values it was built from (its children) |
| `_op` | which operation made it, like `"*"` |
| `_backward` | a function that passes this Value's grad on to its children |

Running the arithmetic left to right to get o is the **forward pass**. Filling in every `grad`, right to left, is the **backward pass**, also called **backpropagation**.

## Backprop by hand {#by-hand}
> Start with the output's grad set to 1, then let each operation multiply the grad it received by its local derivatives and add the result to its inputs' grads.

Walk the neuron backwards. The output's slope with respect to itself is 1, so o.grad = 1.

```
tanh: local = 1 − o²
           = 1 − 0.0600 = 0.9400
      n.grad  = 0.9400 × 1 = 0.9400

+:    local = 1 for both inputs
      xw.grad = 0.9400
      b.grad  = 0.9400

*:    local for x is w, for w is x
      x.grad = −0.5 × 0.94 = −0.47
      w.grad =  2.0 × 0.94 =  1.88
```

These are the numbers step 1 prints. Read x.grad = −0.47 as: nudge x up a little and o goes *down*, at about half the rate. That's because w is negative.

Two classic mistakes the tests look for:

- **Using the input instead of the output in tanh.** The rule is 1 − tanh(x)², and tanh(x) is the output, o (here −0.2449), not the input n.
- **Using `.grad` where you need `.data`.** The local derivative of a product is the other input's *value*. Its `grad` is a different number (and still 0 at that moment).

## Why grads add up {#accumulate}
> A Value used in two places receives gradient from both, so `_backward` must add to `grad` with `+=`, never overwrite it with `=`.

Take b = a + a with a = 3. Nudge a by 0.001 and b moves by 0.002, so the slope is 2. In the graph, a feeds the `+` twice, and each use sends back 1. Only adding them gives 2; with `=`, the second use overwrites the first and you get 1.

This happens all the time in real models: one weight is used for every example in a batch, and one character's vector is used every time that character appears.

The flip side: grads also add up **across** calls to backward. A training loop sets every grad back to 0 before each backward pass, otherwise the slopes from the previous step leak into the next one.

## The right order: topological sort {#order}
> Each Value's `_backward` may only run after every Value that uses it has finished, so backward walks the graph from the output back, in reverse dependency order.

A Value's grad is complete only after everything that uses it has sent its share. In the diagram, n's grad is ready only after tanh has run, and x's only after `*` has run.

Ordering things so each comes after everything it depends on is called a **topological sort**. You've met it if you've used a build tool: Maven or npm builds a library's dependencies before the library itself. Here the "dependencies" of a Value are the Values it was built from.

The usual way to build that list is a **depth-first search**: to place a Value, first place everything it was built from, then append the Value itself. Reverse the list and you have backward's order: output first, inputs last. Two details matter, and the step 2 tests check both:

- **Remember who you've visited.** In the neuron every path is unique, but in a + a or in bigger graphs a Value can be reached along several paths. Without a visited set it gets listed, and its `_backward` run, more than once.
- **Append after the children, not before.** Appending first puts a Value before its own inputs, so backward would run it too late.

## More operations, and checking against PyTorch {#more-ops}
> Each new operation only needs its forward rule and its local derivative. Build tanh from exp, powers and division, and the gradients come out the same.

tanh can be written with exponentials:

```
tanh(x) = (e^2x − 1) / (e^2x + 1)
```

Step 3 adds two operations, power (`x ** k`) and `exp`, and defines division as multiplying by a power: a / b = a · b⁻¹. Their local derivatives come from the rules table: the power rule, and "eˣ is its own slope". A frequent slip with the power rule is lowering the power but forgetting to bring the old power down as a factor, or the reverse.

To check your engine, the challenges compute the same expression with PyTorch:

```python
x = torch.tensor(
    2.0,
    dtype=torch.float64,
    requires_grad=True)
```

- `requires_grad=True` tells PyTorch to track this tensor, like a micrograd Value.
- `o.backward()` runs the backward pass; afterwards `x.grad` holds the slope.
- `float64` is a 64-bit float. Python's own floats are 64-bit too, so the two engines should agree to the last digit.

The step prints the largest difference between your grads and PyTorch's. That's your quality metric: anything under 0.000001 passes, and a correct engine prints 0.0. The systems metric counts operations with their own `_backward`: +, *, tanh, ** and exp make 5. Minus, negation and division are built from those.

## From gradients to learning {#learning}
> Once every parameter has its grad, take a small step against it. That one move, repeated, is how every model in this course learns.

A positive grad means "increasing this number increases the loss", so you decrease it, and vice versa. Every parameter moves by a small multiple of its grad, in the opposite direction:

```
p.data += -0.05 * p.grad
```

The multiplier, here 0.05, is the **learning rate**. Too small and training crawls; too big and it overshoots and the loss jumps around. One step of training is: forward pass, loss, backward pass, update. The reorder card asks you to put those in order. Lesson 0.3 runs this loop for real, with PyTorch's autograd in place of yours.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Parameter (weight)**: an adjustable number in a model. Training changes parameters.
- **Derivative / slope**: how much an output changes per unit change of an input, for a tiny change.
- **Local derivative**: one operation's slope with respect to one of its own inputs.
- **Gradient (grad)**: the slope of the final output (usually the loss) with respect to a value. Stored in `.grad`.
- **Chain rule**: slopes along a path multiply: dL/da = dL/dc × dc/da.
- **Forward pass**: computing the output from the inputs.
- **Backward pass / backpropagation**: computing every grad, from the output back to the inputs.
- **Autograd**: software that does the backward pass for you, like micrograd or PyTorch.
- **Value**: micrograd's number that remembers how it was made.
- **Graph**: the network of Values and the operations connecting them.
- **Topological sort**: an order where everything comes after what it depends on.
- **Depth-first search (DFS)**: exploring a graph by following each path to the end before backing up.
- **tanh**: a squashing function that maps any number into −1…1. Its slope is 1 − tanh².
- **Neuron**: inputs times weights, plus a bias, through a squashing function: tanh(x·w + b).
- **Bias**: the added number, b, in a neuron.
- **`requires_grad`**: PyTorch's flag to track a tensor for autograd.
- **float64**: 64-bit floats, the same precision as Python's own numbers.
- **Learning rate**: the size of the step against the gradient.
- **Max diff**: the largest gap between your grads and PyTorch's; this lesson's quality metric.
