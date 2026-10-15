import re

import harness
import numpy as np
import torch
import torch.nn.functional as F
from harness import test, fail, skip, learner_fn, arr, script_error, stdout

# Every size different, so a transpose or a wrong sum dim shows up as a wrong shape.
N, D_IN, D_H, V = 5, 3, 4, 6
SHAPES = {"logits": [N, V], "W2": [D_H, V], "b2": [V], "h": [N, D_H], "hpreact": [N, D_H], "W1": [D_IN, D_H], "b1": [D_H]}
# What each gradient is built from, to explain a shape error on its line.
RECIPES = {
    "dW2": f"h is [{N}, {D_H}] and dlogits is [{N}, {V}]. Which product of the two, with one of them transposed, is [{D_H}, {V}]?",
    "dh": f"dlogits is [{N}, {V}] and W2 is [{D_H}, {V}]. Which product of the two, with one of them transposed, is [{N}, {D_H}]?",
    "dW1": f"X is [{N}, {D_IN}] and dhpreact is [{N}, {D_H}]. Which product of the two, with one of them transposed, is [{D_IN}, {D_H}]?",
    "dhpreact": f"1 − h² and dh are both [{N}, {D_H}]. tanh acts on each number separately, so its backward is elementwise too: `*`, not `@`.",
}


def case():
    g = torch.Generator().manual_seed(7)
    X = torch.randn((N, D_IN), generator=g)
    Y = torch.tensor([0, 3, 5, 3, 1])
    W1 = torch.randn((D_IN, D_H), generator=g) * 0.5
    b1 = torch.randn(D_H, generator=g) * 0.1
    W2 = torch.randn((D_H, V), generator=g) * 0.5
    b2 = torch.randn(V, generator=g) * 0.1
    return X, Y, W1, b1, W2, b2


def reference():
    X, Y, *params = case()
    W1, b1, W2, b2 = (p.clone().requires_grad_() for p in params)
    hpreact = X @ W1 + b1
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    for t in (hpreact, h, logits):
        t.retain_grad()
    F.cross_entropy(logits, Y).backward()
    ts = {"logits": logits, "W2": W2, "b2": b2, "h": h, "hpreact": hpreact, "W1": W1, "b1": b1}
    return {k: arr(t.grad).astype(np.float64) for k, t in ts.items()}, h.detach(), hpreact.detach()


def learner_grads(L):
    learner_fn(L, "manual_backward")
    try:
        out = L["manual_backward"](*case())
    except Exception as exc:  # noqa: BLE001 - explain the learner's error in plain words
        info = harness.explain(exc, harness._CTX.get("code", ""), harness._CTX.get("filename", "main.py"))
        src = harness._CTX.get("code", "").splitlines()
        text = src[info["line"] - 1] if info.get("line") and 0 < info["line"] <= len(src) else ""
        m = re.match(r"\s*(d\w+)\s*=", text)
        var = m.group(1) if m else None
        if "cannot be multiplied" in info["message"] and var in RECIPES:
            fail(f"Line {info['line']}: the shapes don't fit `@` ({info['message']}). `{var}` must have the shape of "
                 f"`{var[1:]}`, {SHAPES[var[1:]]}. {RECIPES[var]}")
        where = f" (line {info['line']})" if info.get("line") else ""
        fail(info["diagnosis"] if info["plain"] else f"Calling `manual_backward` raised an error{where}. {info['diagnosis']}")
    if not isinstance(out, dict):
        fail(f"`manual_backward` should return the dict of gradients, but it returns {type(out).__name__}.")
    return out


def grads_or_skip(L):
    try:
        return learner_grads(L)
    except Exception:  # noqa: BLE001 - the "runs" test above already explains it
        skip("Fix the test above first: manual_backward has to run.")


def grad(out, ref, key):
    """The learner's gradient for `key` as a float64 array with the reference's shape."""
    v = out.get(key)
    if v is Ellipsis:
        fail(f"`d{key}` is still `...`, the placeholder.")
    a = arr(v)
    if a is None:
        fail(f"`d{key}` should be a tensor, but it's {type(v).__name__}.")
    want = ref[key]
    if a.shape != want.shape:
        if a.size == want.size and a.squeeze().shape == want.squeeze().shape:
            return a.reshape(want.shape).astype(np.float64)  # e.g. sum(0, keepdim=True): same numbers, one extra dim
        if a.ndim == 2 and a.shape == want.shape[::-1]:
            fail(f"`d{key}` has shape {list(a.shape)}, the transpose of `{key}`'s {SHAPES[key]}. "
                 "A gradient always has the shape of the thing it's the gradient of. Swap the order of the product.")
        fail(f"`d{key}` has shape {list(a.shape)}, but `{key}` is {SHAPES[key]}, and a gradient always has the shape of the thing it's the gradient of.")
    return a.astype(np.float64)


def close(a, b):
    return np.allclose(a, b, rtol=1e-4, atol=1e-6)


def earlier_ok(out, ref, keys):
    try:
        return all(close(grad(out, ref, k), ref[k]) for k in keys)
    except Exception:  # noqa: BLE001 - includes the failures grad() raises
        return False


@test("runs", "manual_backward runs and returns its gradients")
def _(L):
    learner_grads(L)


@test("dlogits", "dlogits: the gradient of cross_entropy")
def _(L):
    out = grads_or_skip(L)
    ref, _h, _hp = reference()
    a = grad(out, ref, "logits")
    want = ref["logits"]
    if close(a, want):
        return
    if close(a, want * N):
        fail(f"Every entry is {N} times too big. `cross_entropy` averages the loss over the {N} rows, "
             "so each row's share of the gradient carries a factor of 1/n.")
    if close(a, -want):
        fail("The sign is flipped. Raising the correct class's logit *lowers* the loss, so its gradient must be negative.")
    probs = want + np.eye(V)[[0, 3, 5, 3, 1]] / N
    if close(a, probs):
        fail("That's softmax / n, with nothing taken away for the correct class. The loss is −log p_y, "
             "so in each row the correct class's entry gets 1 subtracted: d loss / d logit_j = (p_j − [j = y]) / n.")
    fail(f"`dlogits` is off by up to {np.abs(a - want).max():.2e} from autograd's answer.")


@test("layer2", "dW2 and db2: the second layer's weights and bias")
def _(L):
    out = grads_or_skip(L)
    ref, _h, _hp = reference()
    if not earlier_ok(out, ref, ["logits"]):
        skip("Fix dlogits first. Everything downstream is built from it.")
    a = grad(out, ref, "W2")
    if not close(a, ref["W2"]):
        fail(f"`dW2` is off by up to {np.abs(a - ref['W2']).max():.2e}. logits = h @ W2, so each weight's gradient pairs h with dlogits.")
    v = out.get("b2")
    b = arr(v) if v is not Ellipsis else None
    if b is not None and b.shape == (N,):
        fail(f"`db2` has shape [{N}], one number per example, but b2 is [{V}]: you summed over the classes. "
             "b2 was added to every row of the batch, so its gradient adds up over the batch. Which dim is that?")
    b = grad(out, ref, "b2")
    if close(b, ref["b2"] / N):
        fail(f"`db2` is {N} times too small. b2 is added to every one of the {N} rows, so its gradient is the *sum* "
             "of what each row sends back, not their average.")
    if not close(b, ref["b2"]):
        fail(f"`db2` is off by up to {np.abs(b - ref['b2']).max():.2e} from autograd's answer.")


@test("dh", "dh: back through W2 to the hidden layer")
def _(L):
    out = grads_or_skip(L)
    ref, _h, _hp = reference()
    if not earlier_ok(out, ref, ["logits"]):
        skip("Fix dlogits first.")
    a = grad(out, ref, "h")
    if not close(a, ref["h"]):
        fail(f"`dh` is off by up to {np.abs(a - ref['h']).max():.2e}. logits = h @ W2, so dh multiplies dlogits by W2, transposed so the shapes fit.")


@test("through_tanh", "dhpreact: back through tanh")
def _(L):
    out = grads_or_skip(L)
    ref, h, hpreact = reference()
    if not earlier_ok(out, ref, ["logits", "h"]):
        skip("Fix dlogits and dh first.")
    a = grad(out, ref, "hpreact")
    want = ref["hpreact"]
    if close(a, want):
        return
    hh, hp, dh = arr(h).astype(np.float64), arr(hpreact).astype(np.float64), ref["h"]
    if close(a, (1 - hp**2) * dh):
        fail("You used hpreact, tanh's input, in 1 − x². The derivative of tanh is 1 − tanh(x)², and tanh(hpreact) is already in h.")
    if close(a, 1 - hh**2):
        fail("That's tanh's local derivative on its own. The chain rule multiplies it by the gradient arriving from above, dh.")
    fail(f"`dhpreact` is off by up to {np.abs(a - want).max():.2e}. h = tanh(hpreact), and tanh's slope is 1 − h².")


@test("layer1", "dW1 and db1: the first layer")
def _(L):
    out = grads_or_skip(L)
    ref, _h, _hp = reference()
    if not earlier_ok(out, ref, ["logits", "h", "hpreact"]):
        skip("Fix the tests above first.")
    for key in ("W1", "b1"):
        a = grad(out, ref, key)
        if not close(a, ref[key]):
            fail(f"`d{key}` is off by up to {np.abs(a - ref[key]).max():.2e}. The first layer has the same form as the second: "
                 "hpreact = X @ W1 + b1. Reuse the pattern from dW2 and db2.")


@test("script_runs", "The comparison with autograd runs and prints")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "maxdiff" not in stdout():
        fail("Nothing printed the comparison. Keep the loop at the bottom that prints each gradient's maxdiff.")
