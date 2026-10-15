import numpy as np
import torch
from harness import test, fail, warn, learner_fn, arr

# The lesson's square table. Square on purpose: the classic keepdim bug runs without an error here.
N4 = [[3, 1, 0, 4],
      [0, 2, 2, 0],
      [1, 1, 6, 0],
      [5, 0, 0, 5]]
# Not square, with float counts.
N25 = [[1.0, 2.0, 3.0, 0.0, 4.0],
       [0.5, 0.5, 0.0, 2.0, 1.0]]


def check_step1(L):
    fn = L.get("probs_of")
    try:
        ok = callable(fn) and torch.equal(fn(torch.eye(3), torch.tensor([0, 1]), torch.tensor([0, 1])), torch.ones(2))
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: probs_of no longer passes step 2's tests")


def run_on(L, n):
    out = learner_fn(L, "normalize_rows")(torch.tensor(n))
    if out is Ellipsis:
        fail("`normalize_rows` still returns `...`, the placeholder.")
    a = arr(out)
    if a is None:
        fail(f"`normalize_rows` should return a tensor, but it returns {type(out).__name__}.")
    return a


@test("returns_floats", "Returns a float table with N's shape")
def _(L):
    check_step1(L)
    a = run_on(L, N4)
    if a.dtype.kind != "f":
        x = np.array(N4)
        if np.array_equal(a, x // x.sum(1, keepdims=True)):
            fail("Every entry came out as a whole number, mostly 0. `//` is floor division: 3 // 8 is 0. Which operator keeps the fraction?")
        fail(f"The result is {a.dtype.name}, so it can only hold whole numbers. Probabilities need floats.")
    if a.shape != (4, 4):
        fail(f"The result has shape {list(a.shape)}, but N is [4, 4]. Return one probability per count.")


@test("rows_sum_to_one", "Every row sums to 1")
def _(L):
    a = run_on(L, N4)
    x = np.array(N4, dtype=np.float32)
    rows = a.sum(1)
    if np.allclose(rows, 1, atol=1e-5):
        return
    if np.allclose(a, x / x.sum(1)[None, :], atol=1e-6):
        fail("Rows don't sum to 1, even though no error appeared. `N.sum(1)` has shape [4], and broadcasting lines a [4] up with the "
             "*last* dimension, so entry [i, j] was divided by row j's total instead of row i's. "
             "What shape does the column of row totals need, so each row meets its own total?")
    if np.allclose(a.sum(0), 1, atol=1e-5):
        fail("Your *columns* sum to 1, not your rows. `dim=0` adds down each column. Which dim runs along a row?")
    if np.allclose(a, x / x.sum(), atol=1e-6):
        fail(f"Every entry was divided by the grand total, {int(x.sum())}, so the whole table sums to 1 instead of each row. "
             "Each row needs its own total.")
    if np.isinf(a).any() or (a > 1).any():
        fail("Some entries are above 1 (or infinite), and a probability can't be. Is the division the right way up?")
    fail(f"Row sums are {np.round(rows, 3).tolist()}. Each should be 1.")


@test("any_shape", "Works on a table that isn't square")
def _(L):
    a = run_on(L, N25)
    x = np.array(N25, dtype=np.float32)
    if a.shape != x.shape or not np.allclose(a, x / x.sum(1, keepdims=True), atol=1e-6):
        fail("On a [2, 5] table of float counts the rows don't come out right. Check that nothing depends on N being 4×4.")
