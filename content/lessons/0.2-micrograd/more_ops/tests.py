import math

import harness
import torch
from harness import test, fail, warn, script_error, stdout


class _Placeholder(Exception):
    pass


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


def value_class(L):
    V = L.get("Value")
    if V is None or not hasattr(V, "__pow__") or not hasattr(V, "exp"):
        err = script_error()
        if err:
            fail(f"`Value.__pow__` and `Value.exp` never got attached because your script stopped first: {err['diagnosis']}")
        fail("`Value` has no `__pow__` or `exp`. Keep the lines that attach `value_pow` and `value_exp` to the class.")
    return V


def check_topo(L):
    V = L["Value"]
    try:
        a = V(2.0)
        y = a * a + a
        y.backward()
        ok = abs(a.grad - 5.0) < 1e-9
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: backward() no longer passes step 2's tests")


def grad_after(V, x0, f, upstream=2.0):
    x = V(x0)
    y = run(lambda: f(x), "The forward pass")
    y.grad = upstream
    run(y._backward, "`_backward`")
    return y, x.grad


@test("pow_grad", "x**k sends k·x^(k−1)·out.grad back")
def _(L):
    check_topo(L)
    V = value_class(L)
    for k in (2, 3, -1, 0.5):
        try:
            y, g = grad_after(V, 4.0, lambda x: x**k)
        except _Placeholder:
            fail("In `value_pow`, `d_pow` is still `...`, the placeholder. How fast does x**k grow as x moves?")
        want = k * 4.0 ** (k - 1) * 2.0
        if abs(g - want) < 1e-9:
            continue
        story = f"With x = 4, k = {k} and out.grad = 2, x.grad is {g:g} instead of {want:g}. "
        if abs(g - k * 4.0**k * 2.0) < 1e-9:
            fail(story + "The power didn't come down by one: the slope of x**k is k times x to the power k − 1.")
        if abs(g - 4.0 ** (k - 1) * 2.0) < 1e-9:
            fail(story + "The power dropped by one, but the old power didn't come out in front as a factor. What's the slope of x² at x = 4?")
        fail(story + "The local derivative of x**k is k·x**(k − 1).")


@test("exp_grad", "exp sends e^x·out.grad back")
def _(L):
    V = value_class(L)
    try:
        y, g = grad_after(V, 0.5, lambda x: x.exp())
    except _Placeholder:
        fail("In `value_exp`, `d_exp` is still `...`, the placeholder. What's special about the slope of e**x?")
    want = math.exp(0.5) * 2.0
    if abs(g - want) < 1e-9:
        return
    story = f"With x = 0.5 and out.grad = 2, x.grad is {g:.4f} instead of {want:.4f}. "
    if abs(g - 0.5 * 2.0) < 1e-9:
        fail(story + "You used x itself as the slope. e**x is its own derivative, and `out.data` already holds e**x.")
    if abs(g - 2.0) < 1e-9:
        fail(story + "A slope of 1 is the rule for +. e**x is its own derivative. Which value holds e**x here?")
    fail(story + "The local derivative of e**x is e**x itself.")


@test("matches_torch", "Expressions built from exp, ** and / match PyTorch")
def _(L):
    V = value_class(L)
    vals = (1.5, -2.0, 0.25)
    a, b, w = (V(x) for x in vals)
    try:
        f = run(lambda: (a / b + (w * a).exp()) ** 2 - b**3, "Building f")
        run(f.backward, "`f.backward()`")
    except _Placeholder:
        fail("A `...` placeholder is still in `value_pow` or `value_exp`.")
    ta, tb, tw = (torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in vals)
    tf = (ta / tb + torch.exp(tw * ta)) ** 2 - tb**3
    tf.backward()
    for name, mine, ref in (("a", a.grad, ta.grad.item()), ("b", b.grad, tb.grad.item()), ("w", w.grad, tw.grad.item())):
        if abs(mine - ref) > 1e-9:
            fail(f"On f = (a/b + e^(w·a))² − b³, your {name}.grad is {mine:.6f} but PyTorch says {ref:.6f}.")


@test("script_runs", "tanh from exp matches PyTorch at the bottom")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "tanh_from_exp" not in L or "o" not in L:
        fail("The tanh-from-exp comparison at the bottom is missing. Put it back.")
    if "max diff" not in stdout():
        fail("Nothing printed the max diff. Add the print lines back so you can see the comparison.")
