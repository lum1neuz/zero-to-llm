import math

import numpy as np
import torch
import harness
from harness import test, fail, learner_fn, arr, as_float, script_error

XS = [0, 5, 13, 13, 1, 0]
YS = [5, 13, 13, 1, 0, 9]


def test_w():
    return torch.randn((27, 27), generator=torch.Generator().manual_seed(99))


def softmax_np(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def call(L, name, *args):
    """Call a learner function. Returns (result, exception) so a test can diagnose the exception."""
    learner_fn(L, name)
    try:
        return L[name](*args), None
    except Exception as exc:  # noqa: BLE001
        return None, exc


@test("softmax_rows", "forward turns logits into rows of probabilities")
def _(L):
    W = test_w()
    out = learner_fn(L, "forward")(W, torch.tensor(XS))
    if out is Ellipsis:
        fail("`probs` is still `...`, the placeholder. Turn the logits into probabilities.")
    a = arr(out)
    if a is None:
        fail(f"`forward` should return a tensor, but it returns {type(out).__name__}.")
    a = a.astype(np.float64)
    if a.shape != (len(XS), 27):
        fail(f"`forward` returns shape {list(a.shape)}. For {len(XS)} input characters it should be [{len(XS)}, 27]: one distribution per input.")
    logits = W.numpy().astype(np.float64)[XS]
    want = softmax_np(logits)
    if np.allclose(a, want, atol=1e-5):
        return
    if (a < 0).any():
        fail("Some of your probabilities are negative. Logits can be any number, so dividing them by their sum doesn't give probabilities. "
             "What turns any number into a positive count?")
    rows = a.sum(1)
    if np.allclose(a, np.exp(logits), rtol=1e-4):
        fail(f"Your rows sum to {rows[0]:.2f}, {rows[1]:.2f}, …, not 1. `exp` turns log-counts into positive counts. What turns counts into probabilities?")
    if np.isclose(a.sum(), 1, atol=1e-4):
        fail("The whole table sums to 1 instead of each row. Every input character needs its own distribution over the next one.")
    if np.allclose(a.sum(0), 1, atol=1e-4):
        fail("Your *columns* sum to 1. Each row is one input's distribution over the 27 next characters. Which dim runs along a row?")
    fail(f"Row sums are {np.round(rows, 3).tolist()} and the values don't match softmax(logits). Exponentiate, then divide each row by its own total.")


@test("loss_value", "loss_fn averages −log p of the right next characters")
def _(L):
    g = torch.Generator().manual_seed(5)
    probs = torch.rand((len(YS), 27), generator=g) + 0.05
    probs = probs / probs.sum(1, keepdim=True)
    ys = torch.tensor(YS)
    out, exc = call(L, "loss_fn", probs, ys)
    if exc is not None:
        n = L.get("xs").nelement() if isinstance(L.get("xs"), torch.Tensor) else None
        if isinstance(exc, IndexError) and n and n != len(YS):
            fail(f"`loss_fn` was given {len(YS)} predictions but indexed as if there were {n}, the size of your script's training set. "
                 "Work out the row count from the arguments it's given.")
        learner_fn(L, "loss_fn")(probs, ys)
    if out is Ellipsis:
        fail("`loss_fn` still returns `...`, the placeholder.")
    a = arr(out)
    if a is not None and a.size > 1:
        fail(f"`loss_fn` returns {a.size} numbers. A loss is one number: average them.")
    v = as_float(out)
    if v is None:
        fail(f"`loss_fn` should return a single number, but it returns {type(out).__name__}.")
    p = probs.numpy().astype(np.float64)
    picked = p[np.arange(len(YS)), YS]
    want = float(-np.log(picked).mean())
    if abs(v - want) < 1e-4:
        return
    if abs(v + want) < 1e-4:
        fail(f"Your loss is {v:.4f}, negative. Log-probabilities are negative, so the loss flips their sign: lower should mean better.")
    if abs(v - float(-np.log(p[:, YS]).mean())) < 1e-4:
        fail(f"`probs[:, ys]` takes every row's probability for every target, a {len(YS)}×{len(YS)} table. "
             "Each row should only be scored on its own target. How did you pick one cell per row in 0.0?")
    fail(f"Your loss is {v:.4f}, but the average of −log p over each row's target is {want:.4f}.")


def parts_work(L):
    """True when forward and loss_fn both give the right numbers on a small case."""
    try:
        W = test_w()
        p = L["forward"](W, torch.tensor(XS))
        ok = np.allclose(arr(p), softmax_np(W.numpy().astype(np.float64)[XS]), atol=1e-5)
        v = as_float(L["loss_fn"](p, torch.tensor(YS)))
        want = float(-np.log(arr(p).astype(np.float64)[np.arange(len(YS)), YS]).mean())
        return ok and v is not None and abs(v - want) < 1e-4
    except Exception:  # noqa: BLE001
        return False


@test("trains", "Gradient descent brings the loss down")
def _(L):
    err = script_error()
    if err and "leaf Variable" in err["message"]:
        fail("`W -= …` changes W in place, and W is a leaf that autograd is tracking, so PyTorch refuses. "
             "Update the numbers behind W without recording it: through `W.data`, or inside `with torch.no_grad():`.")
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    loss, W, xs, ys = (L.get(k) for k in ("loss", "W", "xs", "ys"))
    if not all(isinstance(t, torch.Tensor) for t in (loss, W, xs, ys)):
        fail("`loss`, `W`, `xs` or `ys` is missing. Keep the training loop at the bottom of your script.")
    if not parts_work(L):
        harness.skip("Fix the tests above first. Training needs a working forward and loss_fn.")
    W0 = torch.randn((27, 27), generator=torch.Generator().manual_seed(2147483647)).numpy().astype(np.float64)
    x, y = xs.numpy(), ys.numpy()
    start = float(-np.log(softmax_np(W0[x])[np.arange(len(y)), y]).mean())
    final = as_float(loss)
    if np.allclose(W.detach().numpy(), W0, atol=1e-7):
        fail(f"W never changed, so the loss stayed at {start:.4f}. The loop computes `W.grad` every step but never uses it. "
             "Which way should W move to make the loss smaller, and by how much?")
    if final is None or math.isnan(final) or final > start:
        fail(f"The loss went *up*, from {start:.4f} to {final:.4f}. The gradient points uphill. Which way should you step?")
    if final > 2.3:
        fail(f"After 100 steps the loss is {final:.4f}, down from {start:.4f} but not by much. The counts model gets about 2.06 on these names. "
             "Check the learning rate: the stub asks for 50.")
