import math

import harness
from harness import test, fail, script_error


class _Placeholder(Exception):
    pass


def value_class(L):
    V = L.get("Value")
    if V is None:
        err = script_error()
        if err:
            fail(f"`Value` never got defined because your script stopped first: {err['diagnosis']}")
        fail("There's no class called `Value`. The tests build expressions with it. Was it renamed?")
    return V


def run(fn, what):
    """Run learner code; turn a `...` placeholder into _Placeholder and anything else into a plain-words failure."""
    try:
        return fn()
    except (TypeError, AttributeError) as exc:
        if "ellipsis" in str(exc):
            raise _Placeholder() from None
        explain_and_fail(exc, what)
    except Exception as exc:  # noqa: BLE001 - anything the learner's code raises
        explain_and_fail(exc, what)


def explain_and_fail(exc, what):
    info = harness.explain(exc, harness._CTX.get("code", ""), harness._CTX.get("filename", "main.py"))
    where = f" (line {info['line']})" if info.get("line") else ""
    fail(f"{what} raised an error{where}. {info['diagnosis']}")


def num(v):
    return f"{v:g}".replace("-", "−")


@test("mul_grads", "a * b sends b·out.grad to a and a·out.grad to b")
def _(L):
    V = value_class(L)
    a, b = V(3.0), V(-2.0)
    try:
        c = run(lambda: a * b, "`a * b`")
        if abs(c.data - (-6.0)) > 1e-9:
            fail(f"`Value(3.0) * Value(-2.0)` gives data {c.data}, not −6. The forward line in `__mul__` was changed.")
        c.grad = 2.0
        run(c._backward, "`__mul__`'s `_backward`")
    except _Placeholder:
        fail("In `__mul__`, `d_self` or `d_other` is still `...`, the placeholder. "
             "If you nudge `self` up a tiny bit, how much does `self * other` move?")
    ga, gb = a.grad, b.grad
    if abs(ga - (-4.0)) < 1e-9 and abs(gb - 6.0) < 1e-9:
        return
    story = (f"With a = 3, b = −2 and out.grad = 2, you get a.grad = {num(ga)} and b.grad = {num(gb)}. ")
    if abs(ga - 6.0) < 1e-9 and abs(gb + 4.0) < 1e-9:
        fail(story + "They're swapped: each input got its *own* value. Nudge a by h and a·b moves by b·h, so the slope with respect to a is b.")
    if abs(ga - 2.0) < 1e-9 and abs(gb - 2.0) < 1e-9:
        fail(story + "Both just pass out.grad through. That's the rule for +. For a product, how much does a·b move when a moves by h?")
    if abs(ga) < 1e-12 and abs(gb) < 1e-12:
        fail(story + "Both are 0. The local derivative comes from the other input's *value* (`.data`), not its `.grad`, which is still 0 here.")
    if abs(ga - (-2.0)) < 1e-9 and abs(gb - 3.0) < 1e-9:
        fail(story + "That's the local derivative alone. The chain rule multiplies it by out.grad, the gradient arriving from above.")
    fail(story + "They should be −4 (b × out.grad) and 6 (a × out.grad).")


@test("tanh_forward", "tanh computes the hyperbolic tangent")
def _(L):
    V = value_class(L)
    try:
        y = run(lambda: V(0.5).tanh(), "`Value(0.5).tanh()`")
    except _Placeholder:
        fail("In `tanh`, the forward pass hits a `...` placeholder.")
    if y.data is Ellipsis:
        fail("In `tanh`, `t` is still `...`, the placeholder. The `math` module has the function you need.")
    want = math.tanh(0.5)
    if not isinstance(y.data, (int, float)):
        fail(f"`Value(0.5).tanh().data` is a {type(y.data).__name__}. It should be a plain float, like every other Value's data.")
    if abs(y.data - want) < 1e-9:
        return
    if abs(y.data - math.tan(0.5)) < 1e-9:
        fail(f"`Value(0.5).tanh()` gives {y.data:.4f}, which is `math.tan`, the ordinary tangent. You want the *hyperbolic* tangent, {want:.4f}.")
    fail(f"`Value(0.5).tanh()` gives {y.data:.4f}, but tanh(0.5) is {want:.4f}.")


@test("tanh_backward", "tanh sends (1 − t²)·out.grad back")
def _(L):
    V = value_class(L)
    x = V(0.5)
    try:
        y = run(x.tanh, "`Value(0.5).tanh()`")
        y.grad = 2.0
        run(y._backward, "`tanh`'s `_backward`")
    except _Placeholder:
        fail("In `tanh`, `local` is still `...`, the placeholder. What's the slope of tanh, written with its output t?")
    t = math.tanh(0.5)
    want = (1 - t * t) * 2.0
    g = x.grad
    if abs(g - want) < 1e-9:
        return
    story = f"With x = 0.5 and out.grad = 2, x.grad comes out {g:.4f} instead of {want:.4f}. "
    if abs(g - (1 - 0.25) * 2.0) < 1e-9:
        fail(story + "You used the input, 0.5, where the formula needs t = tanh(0.5), the output.")
    if abs(g - (1 - t) * 2.0) < 1e-9:
        fail(story + "Close: the slope of tanh is 1 − t², with t squared.")
    if abs(g - t * 2.0) < 1e-9:
        fail(story + "That uses tanh's value as its slope. The slope is 1 at x = 0 and flattens towards 0 as tanh saturates at ±1. Which expression in t does that?")
    if abs(g - (1 - t * t)) < 1e-9:
        fail(story + "That's the local derivative alone. Multiply it by out.grad, the gradient arriving from above.")
    fail(story + "The local derivative of tanh is 1 − t², times out.grad.")


def ops_work(L):
    V = L["Value"]
    try:
        a, b = V(3.0), V(-2.0)
        c = a * b
        c.grad = 1.0
        c._backward()
        x = V(0.5)
        y = x.tanh()
        y.grad = 1.0
        y._backward()
        t = math.tanh(0.5)
        return abs(a.grad + 2) < 1e-9 and abs(b.grad - 3) < 1e-9 and abs(y.data - t) < 1e-9 and abs(x.grad - (1 - t * t)) < 1e-9
    except Exception:  # noqa: BLE001
        return False


@test("neuron", "The hand backprop at the bottom runs")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    for name in ("x", "w", "b", "o"):
        if L.get(name) is None or not hasattr(L[name], "grad"):
            fail(f"`{name}` is missing from the bottom of your script. Put the neuron example back.")
    t = math.tanh(2.0 * -0.5 + 0.75)
    dn = 1 - t * t
    want = {"x": dn * -0.5, "w": dn * 2.0, "b": dn}
    if any(abs(L[name].grad - g) > 1e-6 for name, g in want.items()) and not ops_work(L):
        harness.skip("Fix the tests above first. The neuron is built from * and tanh.")
    for name, g in want.items():
        if abs(L[name].grad - g) > 1e-6:
            fail(f"`{name}.grad` is {L[name].grad:.4f} after the hand backprop, not {g:.4f}. "
                 "Check the order: each `_backward()` needs its own output's grad to be finished first, so go from o back to x.")
