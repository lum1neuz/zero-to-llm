import math

import harness
import torch
from harness import test, fail, warn, learner_fn, script_error, stdout


def graph(V):
    """e = (a*b + a) * (a*b): c = a*b is used twice and a three times."""
    a, b = V(1.5), V(-2.0)
    c = a * b
    d = c + a
    e = d * c
    return {"a": a, "b": b, "c": c, "d": d, "e": e}


def check_step1(L):
    V = L.get("Value")
    try:
        x = V(0.5)
        y = (x * 2.0).tanh()
        y.grad = 1.0
        y._backward()
        ok = abs(y.data - math.tanh(1.0)) < 1e-9
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: Value's * or tanh no longer passes step 1's tests")


def order_of(L, root):
    out = learner_fn(L, "topo_order")(root)
    if out is None:
        fail("`topo_order` returns nothing. Keep the `return order` line at the end.")
    if not isinstance(out, (list, tuple)):
        fail(f"`topo_order` should return a list of Values, but it returns {type(out).__name__}.")
    return list(out)


def name_of(nodes, v):
    return next((k for k, n in nodes.items() if n is v), "an extra Value")


@test("returns_list", "topo_order returns a list that includes the root")
def _(L):
    check_step1(L)
    V = L["Value"]
    root = V(1.0)
    order = order_of(L, root)
    if not order:
        fail("`topo_order` returns an empty list, so `backward()` has nothing to walk. "
             "Whatever the graph looks like, which Value is sure to be in it?")
    if not any(v is root for v in order):
        fail("The list doesn't contain the root, the Value you called `topo_order` on. It belongs at the very end.")


@test("every_node_once", "Every Value in the graph appears exactly once")
def _(L):
    nodes = graph(L["Value"])
    order = order_of(L, nodes["e"])
    for name, v in nodes.items():
        count = sum(1 for x in order if x is v)
        if count > 1:
            fail(f"For e = (a·b + a) · (a·b), `{name}` appears {count} times. A Value that's used twice gets listed twice, "
                 f"so its `_backward` would run twice and double-count its gradient. How can the search remember who it has already visited?")
    missing = [name for name, v in nodes.items() if not any(x is v for x in order)]
    if missing:
        fail(f"For e = (a·b + a) · (a·b), the list is missing {', '.join(missing)}. It has to reach every Value e depends on: "
             "the inputs of the inputs too, all the way down to the leaves.")


@test("children_first", "Each Value comes after the Values it was built from")
def _(L):
    nodes = graph(L["Value"])
    order = order_of(L, nodes["e"])
    pos = {id(v): i for i, v in enumerate(order)}
    for v in order:
        for child in v._prev:
            if id(child) in pos and pos[id(child)] > pos[id(v)]:
                fail(f"`{name_of(nodes, v)}` comes before `{name_of(nodes, child)}`, which it was built from. "
                     "`backward()` walks the list in reverse, so a Value's `_backward` would run before its own grad is complete. "
                     "When should a Value join the list: before or after everything it depends on?")


@test("matches_torch", "backward() matches PyTorch on an expression that reuses values")
def _(L):
    V = L["Value"]
    vals = (1.5, -2.0, 0.25)
    a, b, w = (V(x) for x in vals)
    c = a * b
    f = ((c + a) * w).tanh() * c - a
    try:
        f.backward()
    except Exception as exc:  # noqa: BLE001
        info = harness.explain(exc, harness._CTX.get("code", ""), harness._CTX.get("filename", "main.py"))
        where = f" (line {info['line']})" if info.get("line") else ""
        fail(f"`backward()` raised an error{where}. {info['diagnosis']}")
    ta, tb, tw = (torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in vals)
    tc = ta * tb
    tf = torch.tanh((tc + ta) * tw) * tc - ta
    tf.backward()
    for name, mine, ref in (("a", a.grad, ta.grad.item()), ("b", b.grad, tb.grad.item()), ("w", w.grad, tw.grad.item())):
        if abs(mine - ref) > 1e-9:
            fail(f"On f = tanh((a·b + a)·w)·(a·b) − a, your {name}.grad is {mine:.6f} but PyTorch says {ref:.6f}.")


@test("script_runs", "The comparison at the bottom runs")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "mine" not in L or "ref" not in L:
        fail("The lines that compare micrograd with PyTorch are missing. Put them back at the bottom.")
    if "max diff" not in stdout():
        fail("Nothing printed the max diff. Add the print lines back so you can see the comparison.")
