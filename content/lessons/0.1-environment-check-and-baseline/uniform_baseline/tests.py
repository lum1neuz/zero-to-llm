import math

import numpy as np
import torch
from harness import test, fail, learner_fn, arr, as_float

LN27 = math.log(27)
UNEVEN = [0.5, 0.25, 0.125, 0.9]


def loss_of(L, values):
    out = learner_fn(L, "mean_nll")(torch.tensor(values))
    if out is Ellipsis:
        fail("`mean_nll` still returns `...`, the placeholder.")
    if out is None:
        fail("`mean_nll` returns nothing. Did the `return` go missing?")
    a = arr(out)
    if a is not None and a.size > 1:
        fail(f"`mean_nll` returns {a.size} numbers, one per prediction. A loss is a single number. How do you combine them?")
    v = as_float(out)
    if v is None:
        fail(f"`mean_nll` should return a single number (a 0-d tensor), but it returns {type(out).__name__}.")
    return v


@test("returns_number", "mean_nll returns one number")
def _(L):
    loss_of(L, [1 / 27] * 8)


@test("uniform_is_ln27", "The uniform model scores ln 27 ≈ 3.2958 nats")
def _(L):
    n = 32
    v = loss_of(L, [1 / 27] * n)
    if abs(v - LN27) < 1e-4:
        return
    if abs(v + LN27) < 1e-4:
        fail(f"Your loss is {v:.4f}. A probability is at most 1, so its log is at most 0, and the average log is negative. "
             "A loss should go *down* as the model gets better. What flips it?")
    if abs(v - math.log2(27)) < 1e-4:
        fail(f"Your loss is {v:.4f}. That's in bits, from log base 2. This course measures loss in nats, with the natural log.")
    if abs(v - math.log10(27)) < 1e-4:
        fail(f"Your loss is {v:.4f}. That's log base 10. This course measures loss in nats, with the natural log.")
    if abs(v - n * LN27) < 1e-2:
        fail(f"Your loss is {v:.1f}, which is {n} × 3.2958. That's the total over all {n} predictions. "
             "If you doubled the data, would the model be twice as bad?")
    if abs(v + 1 / 27) < 1e-5:
        fail(f"Your loss is {v:.4f}, minus the average probability. The loss is built from the *log* of each probability.")
    fail(f"Your loss is {v:.4f}, but a model that gives every one of 27 symbols the same chance should score ln 27 ≈ {LN27:.4f}.")


@test("average_of_logs", "Averages the logs, so each bad guess counts")
def _(L):
    v = loss_of(L, UNEVEN)
    x = np.array(UNEVEN, dtype=np.float64)
    want = float(-np.log(x).mean())
    if abs(v - want) < 1e-4:
        return
    if abs(v + math.log(x.mean())) < 1e-4:
        fail(f"On probabilities {UNEVEN} you get {v:.4f}, minus the log of their average. The right answer is {want:.4f}, "
             "the average of their logs. With the log of the average, one confident wrong guess hides behind the good ones.")
    fail(f"On probabilities {UNEVEN} you get {v:.4f}, but the average negative log-likelihood is {want:.4f}.")
