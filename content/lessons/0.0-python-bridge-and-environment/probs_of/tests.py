import numpy as np
import torch
from harness import test, fail, learner_fn, arr

# Not square, so a swapped or table-shaped answer can't pass by accident.
P35 = [[0.10, 0.20, 0.30, 0.15, 0.25],
       [0.50, 0.05, 0.05, 0.30, 0.10],
       [0.02, 0.08, 0.40, 0.20, 0.30]]
XS = [0, 2, 1, 2]
YS = [2, 0, 1, 2]
# Square, with every cell different, for telling rows from columns.
P44 = [[0.10, 0.20, 0.30, 0.40],
       [0.05, 0.15, 0.35, 0.45],
       [0.12, 0.18, 0.22, 0.48],
       [0.33, 0.27, 0.24, 0.16]]
XS4 = [0, 3, 1, 2]
YS4 = [2, 0, 3, 1]


def run_on(L, p, xs, ys):
    out = learner_fn(L, "probs_of")(torch.tensor(p), torch.tensor(xs), torch.tensor(ys))
    if out is Ellipsis:
        fail("`probs_of` still returns `...`, the placeholder.")
    if out is None:
        fail("`probs_of` returns nothing. Did the `return` go missing?")
    if isinstance(out, list):
        fail(f"`probs_of` returns a Python list of {len(out)} tensors. Tensor code keeps results in one tensor: "
             "`torch.stack(...)` joins a list, and tensor indexing can pick every pair in one go without a loop.")
    a = arr(out)
    if a is None:
        fail(f"`probs_of` should return a tensor, but it returns {type(out).__name__}.")
    return a


def shape(a):
    return "[" + ", ".join(str(d) for d in a.shape) + "]"


@test("returns_tensor", "probs_of returns a tensor")
def _(L):
    run_on(L, P35, XS, YS)


@test("one_per_pair", "One number per (x, y) pair")
def _(L):
    a = run_on(L, P35, XS, YS)
    n = len(XS)
    if a.shape == (n,):
        return
    if a.shape == (n, n):
        fail(f"`probs_of` returns shape {shape(a)}, a whole table. `P[xs]` picks {n} rows, then `[:, ys]` takes "
             f"{n} columns from *every* one of those rows. How do you index rows and columns together, so xs[i] pairs only with ys[i]?")
    if a.shape == (n, len(P35[0])):
        fail(f"`probs_of` returns shape {shape(a)}: one whole row of P per pair. You picked the rows. Which column of each row do you want?")
    fail(f"`probs_of` returns shape {shape(a)}, but 4 pairs should give 4 numbers, shape [4].")


@test("right_values", "Each number is P[xs[i], ys[i]]")
def _(L):
    for p, xs, ys in ((P44, XS4, YS4), (P35, XS, YS)):
        a = run_on(L, p, xs, ys)
        P = np.array(p, dtype=np.float32)
        want = P[xs, ys]
        if a.shape == want.shape and np.allclose(a, want):
            continue
        if a.shape == want.shape and np.allclose(a, P[ys, xs]):
            fail("You read row ys[i], column xs[i], the transpose. In P, the row is the current character (xs) and the column is the next one (ys).")
        fail(f"For the first pair, row {xs[0]} and column {ys[0]}, you return {float(a.reshape(-1)[0]):.2f} but P holds {float(want[0]):.2f} there.")
