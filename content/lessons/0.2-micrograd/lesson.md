---
id: "0.2"
title: micrograd
summary: Build a tiny autograd engine on single numbers, Value by Value, and check every gradient against PyTorch.
phase: "0"
hours: 3
type: build
status: ready
protected: false
cut: null
prerequisites:
  - "0.1"
papers: []
quality_metric:
  id: grad_check_err
  label: Max gradient-check error vs PyTorch
  unit: abs
  note: The max diff printed at the end of step 3, tanh rebuilt from exp against torch.tanh in float64.
  inputs:
    - { id: grad_check_err, label: Max diff vs PyTorch, hint: The last line step 3 prints }
  compute: value
  reference: 0
  pass: { max: 1.0e-6 }
systems_metric:
  id: ops_supported
  label: Ops with backward implemented
  unit: count
  note: "Ops with their own _backward: +, *, tanh, ** and exp after this lesson. Minus, negation and / are built from them."
  inputs:
    - { id: ops_supported, label: Ops with their own _backward }
  compute: value
  reference: 5
  pass: { min: 5 }
quiz:
  - id: plus_equals
    q: "In every `_backward`, why `self.grad += …` and not `self.grad = …`?"
    answer: 0
    options:
      - text: A Value used in two places gets gradient from both, and the two have to add up
        why: "In `b = a + a`, a.grad must be 2. With `=`, the second use overwrites the first and a.grad comes out 1. Karpathy hits exactly this bug halfway through the micrograd video."
      - text: It's faster than assigning
        why: Speed is the same. The difference is correctness when a Value is used more than once.
      - text: So gradients carry over from one training step to the next
        why: "That carry-over is a side effect you *don't* want. It's why a training loop zeroes every grad before calling backward()."
  - id: reverse_topo
    q: Why does backward() call each `_backward` in reverse topological order?
    answer: 0
    options:
      - text: So every Value's grad is complete before its `_backward` passes it on
        why: "A Value's grad is the sum over everything that uses it. Reverse topological order runs every user first, so by the time a Value sends its grad down, nothing more will be added to it."
      - text: Because iterating a list in reverse is faster in Python
        why: The order is about correctness. A Value that passes its grad on too early sends an incomplete number.
      - text: So the leaves, the inputs and weights, go first
        why: "It's the opposite. The output goes first, with grad 1, and the leaves come last."
---

## Goal

Write micrograd, Karpathy's autograd engine for single numbers, one piece at a time. A `Value` wraps a number and remembers the operation that made it. Each operation knows its own local derivative, and `backward()` chains them from the output back to every input. By the end, a neuron's gradients from your engine match PyTorch's to the last digit.

## Learn

```card
type: concept
id: nudge_and_chain
eyebrow: The whole idea
title: A gradient says which way to turn each knob, and how hard.
body: |
  Nudge an input up a tiny bit and watch the output. With `c = a * b`, a = 2 and b = −3, nudging a up by 0.001 moves c by −0.003: three times the nudge, the other way. That number, −3, is the **gradient** of c with respect to a. For a product it's always the other input, b.

  When steps form a chain, the nudges multiply. If c changes 3 for every unit a changes, and the loss changes 2 for every unit c changes, the loss changes 2 × 3 = 6 for every unit a changes. That's the **chain rule**.

  micrograd keeps each number's gradient in `.grad`. `backward()` starts at the output with a gradient of 1 and passes gradients back, step by step, to every input.
```

```card
type: calc
id: numeric_slope
eyebrow: Do the math
title: What's the slope of f at x = 3?
given:
  - { label: f(x), value: "3x² − 4x + 5" }
  - { label: x, value: "3" }
  - { label: Nudge h, value: "0.0001" }
unit: slope
answer: { min: 13.99, max: 14.01 }
placeholder: "0"
why_ok: "(f(3.0001) − f(3)) / 0.0001 ≈ 14.0003, and the exact slope 6x − 4 is 14 at x = 3. Every _backward you write is this number, worked out by hand instead of by nudging."
why_wrong: "Either nudge it, (f(3 + h) − f(3)) / h, or differentiate: the slope of 3x² − 4x + 5 is 6x − 4."
hint: "f(3) = 20. Work out f(3.0001), or use the power rule: the slope of x² is 2x."
```

```card
type: predict
id: predict_mul_backward
eyebrow: Predict the output
title: What prints?
code: |
  a = Value(2.0)
  b = Value(-3.0)
  c = a * b
  c.grad = 1.0
  c._backward()
  print(a.grad, b.grad)
options:
  - text: "-3.0 2.0"
    correct: true
    why: "Each input's local derivative is the *other* input's value. Nudge a and a·b moves by b = −3; nudge b and it moves by a = 2."
  - text: "2.0 -3.0"
    why: "That gives each input its own value. Nudge a by h and a·b moves by b·h, so a's slope is b."
  - text: "1.0 1.0"
    why: That's the rule for +, which passes out.grad straight through. A product scales it by the other input.
  - text: "-6.0 -6.0"
    why: "−6 is c's value, the forward pass. The gradients are slopes, not the result."
hint: If a goes from 2 to 2.001, how much does a * b change?
```

```card
type: fill_blank
id: tanh_backward_blank
eyebrow: Fill the blanks
title: Finish tanh's backward pass. t is tanh(x).
code: |
  def _backward():
      self.grad [[0]] ([[1]]) * out.grad
tokens: ["+=", "=", "1 - t**2", "t", "1 - t", "t**2"]
answer: ["+=", "1 - t**2"]
tips:
  "=": "If x feeds two operations, both send it gradient. `=` keeps only the last one; `+=` adds them up."
  t: "tanh's value isn't its slope. The slope is 1 at x = 0 and flattens to 0 as tanh saturates at ±1."
  "1 - t": "Close. The derivative of tanh is 1 − tanh², with the square."
  "t**2": "That's large where tanh saturates, but the slope there is nearly 0. Flip it: 1 − t²."
why_ok: "d/dx tanh(x) = 1 − tanh(x)², and `+=` lets gradients from every use of x add up."
why_wrong: "Two things: the slope of tanh written with its output t, and an operator that adds rather than overwrites."
hint: "The slope of tanh is 1 at 0 and goes to 0 at the edges, where t is ±1. Which expression does that?"
```

```card
type: reorder
id: training_step
eyebrow: Put it in order
title: Arrange one step of training a micrograd neuron.
lines:
  - ypred = [n(x) for x in xs]
  - loss = sum((yp - y)**2 for y, yp in zip(ys, ypred))
  - loss.backward()
  - "for p in n.parameters(): p.data += -0.05 * p.grad"
shuffle: [2, 0, 3, 1]
why_ok: "Forward, then the loss, then gradients of the loss, then a small step against each gradient. In a real loop you also zero the grads before backward()."
why_wrong: "Each line needs the one before it: the loss needs predictions, backward() needs a loss, and the update needs gradients."
hint: You can't take a step until you know which way is downhill.
```

```card
type: quiz
ref: plus_equals
```

```card
type: quiz
ref: reverse_topo
```

## Build steps

### Write * and tanh

```challenge
id: local_grads
file: micrograd.py
minutes: 35
start:
  code: |+
    # micrograd: autograd for single numbers
    import math


    class Value:
        """A number that remembers how it was made, so gradients can flow back through it."""

        def __init__(self, data, _children=(), _op=""):
            self.data = data
            self.grad = 0.0
            self._backward = lambda: None  # sends out.grad back to the inputs
            self._prev = set(_children)
            self._op = _op

        def __repr__(self):
            return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

        def __add__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data + other.data, (self, other), "+")

            def _backward():
                # d(out)/d(self) = 1 and d(out)/d(other) = 1
                self.grad += 1.0 * out.grad
                other.grad += 1.0 * out.grad
            out._backward = _backward
            return out

        # Built from + and *, so they need no _backward of their own.
        def __neg__(self):
            return self * -1

        def __sub__(self, other):
            return self + (-other)

        def __radd__(self, other):  # 2 + v
            return self + other

        def __rmul__(self, other):  # 2 * v
            return self * other

        def backward(self):
            # Step 2 writes topo_order. Until then, call each _backward() by hand.
            self.grad = 1.0
            for v in reversed(topo_order(self)):
                v._backward()

setup:
  - say: "Here's Value. It stores data, a grad, the Values it was made from, and a _backward function. Addition is done for you: nudging either input moves the sum by the same amount, so both get out.grad."
    pause: 4.2
  - say: "Your turn, part one: multiplication. The forward pass is written. Fill in the two local derivatives."
    type_file: local_grads/mul_stub.py
    pause: 1.6
  - say: "Part two: tanh, the squashing function in every neuron here. Write its forward pass and its local derivative."
    type_file: local_grads/tanh_stub.py
    pause: 1.6
  - say: "At the bottom, one neuron. It calls each _backward by hand, from the output back. Run the tests when you're ready."
    type_file: local_grads/neuron.py
    pause: 1.0
tests: local_grads/tests.py
solution: local_grads/solution.py
alternatives:
  - local_grads/alternatives/exp_form.py
  - local_grads/alternatives/out_data.py
hints:
  logic: |
    multiplication, out = self × other:
      d_self  = the other input's number
      d_other = self's number
      (each gets multiplied by out.grad for you)
    tanh:
      t = tanh of self's number
        → local slope = 1 − t²   (reuse t, not the input)
        → multiplied by out.grad for you
  tools: |
    read the numbers
      v.data            the plain number stored in a Value v
    tanh of a number
      math.tanh(x)      tanh of a plain number
    the tanh slope
      plain arithmetic: - and ** (power)
  nudge: "For a product, nudge one input and watch the output. For tanh, the derivative is neatest when you write it with tanh's own output."
  concept: "If out = a·b, a tiny change h in a changes out by b·h, so d(out)/da = b and d(out)/db = a. For tanh, d/dx tanh(x) = 1 − tanh(x)², and you already have tanh(x) in t. `math.tanh` computes the forward pass."
  example: |
    Multiplication, with `self.data = 3.0`, `other.data = -2.0`, and `out.grad = 0.5` arriving from above:

    - `out.data` is `3.0 * -2.0 = -6.0`.
    - `d_self` is `-2.0` (the other input) and `d_other` is `3.0`.
    - `self.grad` gets `-2.0 * 0.5 = -1.0`. `other.grad` gets `3.0 * 0.5 = 1.5`.

    tanh, with `self.data = 0.5` and `out.grad = 1.0`:

    - `t` is about `0.4621`.
    - The local derivative is `1 - 0.4621² ≈ 0.7864`.
    - `self.grad` gets `0.7864 * 1.0 = 0.7864`.

    Sanity check: at `self.data = 0.0`, `t` is `0.0` and the slope is exactly `1`. Far from zero, tanh flattens out and the slope shrinks toward `0`.
  skeleton: |
    d_self = _____.data
    d_other = ____.data
    t = math.____(self.data)
    local = 1 - __**2
  solution: |
    d_self = other.data
    d_other = self.data
    t = math.tanh(self.data)
    local = 1 - t**2
on_solve: "x.grad is −0.47: nudge x up and o goes down, because w is negative. Calling every _backward by hand gets old fast, and the order matters. Step 2 automates it."
mistakes:
  - { file: local_grads/mistakes/placeholders.py, fails: mul_grads, says: "`d_self` or `d_other` is still `...`" }
  - { file: local_grads/mistakes/mul_placeholder.py, fails: mul_grads, says: "`d_self` or `d_other` is still `...`" }
  - { file: local_grads/mistakes/mul_swapped.py, fails: mul_grads, says: "They're swapped" }
  - { file: local_grads/mistakes/mul_ones.py, fails: mul_grads, says: "That's the rule for +" }
  - { file: local_grads/mistakes/mul_grads.py, fails: mul_grads, says: "not its `.grad`" }
  - { file: local_grads/mistakes/tanh_placeholder.py, fails: tanh_forward, says: "`t` is still `...`" }
  - { file: local_grads/mistakes/tanh_forward_wrong.py, fails: tanh_forward, says: "the ordinary tangent" }
  - { file: local_grads/mistakes/tanh_of_input.py, fails: tanh_backward, says: "You used the input" }
  - { file: local_grads/mistakes/tanh_no_square.py, fails: tanh_backward, says: "with t squared" }
  - { file: local_grads/mistakes/tanh_sech.py, fails: tanh_backward, says: "uses tanh's value as its slope" }
```

### Automate backward and check against PyTorch

```challenge
id: topo_backward
file: micrograd.py
minutes: 35
start: { from: local_grads }
setup:
  - say: "backward() already sets the output's grad to 1 and walks a list in reverse. What's missing is the list: every Value in the graph, ordered so each one comes after everything it was built from."
    type: "\n\ndef topo_order(root):\n    # TODO: every Value reachable from root, each one listed after all the Values it was built from\n    order = []\n    ...\n    return order\n\n\n"
    pause: 3.0
  - say: "Below it, the same neuron twice: once with your engine, once with PyTorch in float64. Your turn: write topo_order, then compare."
    type_file: topo_backward/compare.py
    pause: 1.2
tests: topo_backward/tests.py
solution: topo_backward/solution.py
alternatives:
  - topo_backward/alternatives/iterative.py
  - topo_backward/alternatives/visited_list.py
hints:
  logic: |
    visited = empty set, order = empty list
    visit(v):
      v already in visited?
        yes → do nothing
        no  → add v to visited
              → visit each child of v
              → only then add v to the end of order
    visit(root)
      → return order
  tools: |
    remember what was visited
      set()             an empty set
      s.add(x)          put x in s
      x in s            is x already in s?
    the inputs of a Value
      v._prev           the Values that v was built from
    go through them, then list v last
      for c in items:   loop over a collection
      lst.append(x)     add x to the end of lst
    a helper that calls itself
      def visit(x):     a function may call itself
  nudge: "A depth-first search does it: before a Value joins the list, everything it was built from has to be in the list already. And one Value can be reached along more than one path."
  concept: "Write a helper that visits a Value: if it hasn't been seen, mark it seen, visit each of its `_prev` children, and *then* append it. Appending after the children puts every Value after its inputs. The seen-set stops a reused Value from being listed twice."
  example: |
    Trace it on `a = 2.0`, `b = 3.0`, `c = a * b`, `d = c + a`. Here `a` is used twice:

    - Start at `d`. Its children are `c` and `a`.
    - Say it visits `c` first: its children are `a` and `b`. Both have no children, so `a` then `b` are appended. Then `c` is appended.
    - Back in `d`: `a` is already visited, so it's skipped. Then `d` is appended.
    - `order` ends as `[a, b, c, d]`. The two children can come out in either order, but `d` is last and `c` comes after both `a` and `b`.

    Why it matters: `backward()` walks that list reversed. `d` pushes 1 into `c` and `a`. Then `c` pushes `b.data = 3.0` more into `a`, so `a.grad = 1 + 3 = 4`, and `b.grad = 2`. If `c` ran before `d`, its `grad` would still be 0 and `a.grad` would stay 1.
  skeleton: |
    visited = ___()

    def build(v):
        if v not in _______:
            visited.___(v)
            for child in v._____:
                build(_____)
            order.______(v)
    build(____)
  solution: |
    visited = set()

    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            order.append(v)
    build(root)
on_solve: "Same numbers as PyTorch. That's all autograd is: a graph, local derivatives, and a topological sort. PyTorch does the same with tensors instead of single numbers."
mistakes:
  - { file: topo_backward/mistakes/placeholder.py, fails: returns_list, says: "returns an empty list" }
  - { file: topo_backward/mistakes/no_visited.py, fails: every_node_once, says: "remember who it has already visited?" }
  - { file: topo_backward/mistakes/one_level.py, fails: every_node_once, says: "the inputs of the inputs too" }
  - { file: topo_backward/mistakes/recursion.py, fails: every_node_once, says: "keeps calling itself" }
  - { file: topo_backward/mistakes/preorder.py, fails: children_first, says: "before or after everything it depends on?" }
```

### Rebuild tanh from exp

```challenge
id: more_ops
file: micrograd.py
minutes: 30
start: { from: topo_backward }
setup:
  - say: "tanh(x) is (e^2x − 1) / (e^2x + 1). Build it from smaller ops and the gradients should come out the same. That needs two more ops, power and exp. Division is x times y to the −1."
    type_file: more_ops/stub.py
    pause: 2.4
  - say: "Your turn: fill in the two local derivatives. The bottom compares tanh-from-exp with PyTorch's tanh."
    pause: 1.0
tests: more_ops/tests.py
solution: more_ops/solution.py
alternatives:
  - more_ops/alternatives/math_exp.py
hints:
  logic: |
    power, out = x raised to k   (x is self's number):
      local slope = k × x raised to (k − 1)
    exp, out = e raised to x:
      local slope = out's own number   (e**x is its own slope)
    both:
      slope × out.grad is added to self.grad   (already written)
  tools: |
    slope of the power op
      v.data            the plain number stored in a Value v
      plain arithmetic: * - and ** (power)
    slope of the exp op
      out.data          the number this op already computed
  nudge: "The power rule for x**k, and e**x, which is its own slope."
  concept: "d/dx x**k = k·x**(k − 1), for any constant k, including −1 and 0.5. d/dx e**x = e**x, and `out.data` already holds e**x, so you don't need to compute it again."
  example: |
    Power, with `self.data = 2.0`, `k = 3`:

    - `out.data` is `2.0**3 = 8.0`.
    - The local derivative is `3 * 2.0**2 = 12.0`.

    Power with `self.data = 4.0`, `k = -1` (this is how division by 4 works):

    - `out.data` is `0.25`.
    - The local derivative is `-1 * 4.0**-2 = -0.0625`. Bigger `x` means smaller `1/x`, so the sign is negative.

    Exp, with `self.data = 1.0`:

    - `out.data` is `e ≈ 2.7183`, and the local derivative is the same `2.7183`.
    - At `self.data = 0.0` both are exactly `1.0`.
  skeleton: |
    d_pow = _ * self.data**(_ - 1)
    d_exp = ___.____
  solution: |
    d_pow = k * self.data**(k - 1)
    d_exp = out.data
on_solve: "Same gradients from one tanh op or from five smaller ones. How finely you split the ops is up to you, as long as each one knows its local derivative. Record the max diff and your op count."
mistakes:
  - { file: more_ops/mistakes/placeholders.py, fails: pow_grad, says: "`d_pow` is still `...`" }
  - { file: more_ops/mistakes/pow_placeholder.py, fails: pow_grad, says: "`d_pow` is still `...`" }
  - { file: more_ops/mistakes/pow_no_drop.py, fails: pow_grad, says: "The power didn't come down by one" }
  - { file: more_ops/mistakes/pow_no_k.py, fails: pow_grad, says: "didn't come out in front" }
  - { file: more_ops/mistakes/exp_placeholder.py, fails: exp_grad, says: "`d_exp` is still `...`" }
  - { file: more_ops/mistakes/exp_input.py, fails: exp_grad, says: "You used x itself as the slope" }
  - { file: more_ops/mistakes/exp_one.py, fails: exp_grad, says: "A slope of 1 is the rule for +" }
```

### Record your metrics

```metrics
id: record
intro: The max diff your engine printed against PyTorch, and how many ops have their own _backward.
```

## Hints

- Follow along with Karpathy's video "The spelled-out intro to neural networks and backpropagation: building micrograd". This lesson is its first half; the second half builds a `Neuron`, `Layer` and `MLP` on top of `Value`.
- Copy `micrograd.py` to your machine and run it with real PyTorch. The comparison lines work unchanged.
- To check any new op, compare its gradient with a nudge: `(f(x + 1e-6) - f(x)) / 1e-6`.
- Add `relu` yourself: forward `0 if x < 0 else x`, local derivative `1 if out > 0 else 0`. It's the activation used in micrograd's MLP.

## Done when

- [ ] `*` and `tanh` send the right gradients back {#local_grads auto=challenge:local_grads}
- [ ] backward() matches PyTorch, including Values used twice {#topo_backward auto=challenge:topo_backward}
- [ ] tanh rebuilt from exp, ** and / matches PyTorch {#more_ops auto=challenge:more_ops}
- [ ] Max diff and op count recorded {#metrics_recorded auto=metrics}
