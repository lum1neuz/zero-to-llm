import numpy as np
import torch
from harness import test, fail, learner_fn, arr, script_error, stdout, LearnerError


# The same helpers the script uses, so the reference U matches the learner's to the last bit.
def inverse(A):
    n = A.shape[0]
    M = torch.cat([A.clone(), torch.eye(n)], dim=1)
    for i in range(n):
        M[i] = M[i] / M[i, i]
        for r in range(n):
            if r != i:
                M[r] = M[r] - M[r, i] * M[i]
    return M[:, n:]


def upper_cholesky(A):
    n = A.shape[0]
    U = torch.zeros_like(A)
    for i in range(n):
        U[i, i] = (A[i, i] - (U[:i, i] ** 2).sum()).sqrt()
        for k in range(i + 1, n):
            U[i, k] = (A[i, k] - (U[:i, i] * U[:i, k]).sum()) / U[i, i]
    return U


def lesson_data():
    g = torch.Generator().manual_seed(6)
    mix = torch.randn(3, 8, generator=g)
    X = torch.randn(64, 3, generator=g) @ mix + 0.1 * torch.randn(64, 8, generator=g)
    X[:, 7] = 0.0
    W = 0.3 * torch.randn(4, 8, generator=g)
    H = 2 * X.T @ X / X.shape[0]
    idx = torch.arange(8)
    H = H + 0.01 * H[idx, idx].mean() * torch.eye(8)
    return X, W, H


X, W, H = lesson_data()
W2 = 0.3 * torch.randn(4, 8, generator=torch.Generator().manual_seed(12))
U = arr(upper_cholesky(inverse(H)))
HINV = arr(inverse(H))


def scale_of(Wt):
    return arr(Wt.abs().amax(dim=1) / 7)


def ref(Wt, M=None, divide=True, sign=1, clamp=True):
    """GPTQ in NumPy float32, with switches for the usual mistakes. M is the matrix whose rows drive the update."""
    M = U if M is None else M
    w = arr(Wt).copy()
    s = scale_of(Wt)
    Q = np.zeros_like(w)
    for j in range(w.shape[1]):
        r = np.round(w[:, j] / s)
        q = (np.clip(r, -7, 7) if clamp else r) * s
        Q[:, j] = q
        err = (w[:, j] - q) / M[j, j] if divide else w[:, j] - q
        w[:, j:] -= sign * err[:, None] * M[j, j:]
    return Q


def rtn(Wt):
    s = scale_of(Wt)[:, None]
    return np.clip(np.round(arr(Wt) / s), -7, 7) * s


def call(L, Wt):
    gptq = learner_fn(L, "gptq")
    try:
        Q = gptq(Wt, H, torch.tensor(scale_of(Wt)))
    except LearnerError as e:
        if "must match the size of tensor" in str(e):
            fail("`err` has one value per row (4 of them) and `U[j, j:]` one per column still to do (8 when j = 0). "
                 "The update needs a value for every weight in `W[:, j:]`, a [4, 8] block. Make `err` a column, "
                 "`err[:, None]`, so the two broadcast into an outer product.")
        raise
    if Q is None:
        fail("`gptq` returns nothing. Keep the `return Q` line at the end.")
    Qa = arr(Q)
    if Qa is None:
        fail(f"`gptq` should return a tensor, but it returns {type(Q).__name__}.")
    if Qa.shape != (4, 8):
        fail(f"`gptq` returns shape [{', '.join(map(str, Qa.shape))}], but W is [4, 8]. Q holds one quantized value per weight.")
    return Qa


@test("on_grid", "Q holds each row's int4 levels, in the weights' units")
def _(L):
    Qa = call(L, W)
    s = scale_of(W)[:, None]
    steps = Qa / s
    if np.allclose(Qa, np.round(Qa)) and np.abs(Qa).max() <= 7 and np.abs(Qa).max() >= 1:
        fail("Q holds the integers themselves (−7…7). Multiply each one back by its row's `scale`, so Q is in the weights' "
             "units and the error `W[:, j] - q` compares like with like.")
    if not np.allclose(steps, np.round(steps), atol=1e-3):
        i, j = np.unravel_index(int(np.argmax(np.abs(steps - np.round(steps)))), steps.shape)
        fail(f"Q[{i}, {j}] is {Qa[i, j]:.4f}, which is {steps[i, j]:.2f} steps of row {i}'s scale: it isn't a whole number of "
             "steps, so it isn't on the int4 grid. Round after dividing by `scale`.")
    if np.abs(steps).max() > 7 + 1e-3:
        fail("Some of Q sits beyond ±7 steps of its row's scale, which int4 can't store. Clamp after rounding.")


@test("compensates", "Each column's rounding error moves onto the columns after it")
def _(L):
    for Wt in (W, W2):
        Qa = call(L, Wt)
        near = lambda v: np.allclose(Qa, v, atol=1e-5)
        if near(ref(Wt)):
            continue
        if near(rtn(Wt)):
            fail("Your Q is plain round-to-nearest: each weight is rounded on its own, and column j's error never reaches the "
                 "columns after it. After quantizing column j, subtract its error, weighted by row j of U, from the columns still to do.")
        if near(ref(Wt, divide=False)):
            fail("The error is never divided by `U[j, j]`, so every correction comes out the wrong size. "
                 "GPTQ's update is (W[:, j] − q) / U[j, j], times row j of U.")
        if near(ref(Wt, sign=-1)):
            fail("Your correction pushes the later columns the same way column j moved, so the errors pile up instead of "
                 "cancelling. The line `W[:, j:] -= update` already subtracts. Should `update` carry an extra minus sign?")
        if near(ref(Wt, M=arr(H))):
            fail("Your update uses the rows of H itself. H says how features move together. The correction comes from its "
                 "inverse, through U, the Cholesky factor of H⁻¹ that the stub already computes.")
        if near(ref(Wt, M=HINV)):
            fail("Your update uses the rows of `inverse(H)` directly. Those are right for the first column, but they go stale: "
                 "once column 0 is fixed, the inverse for the remaining columns changes. U's rows have that bookkeeping built in. Use U.")
        if near(ref(Wt, clamp=False)):
            fail("Without a clamp, compensation pushed some weights past ±7 steps, which int4 can't store. Clamp after rounding.")
        R = ref(Wt)
        j = int(np.flatnonzero(np.abs(Qa - R).max(axis=0) > 1e-5)[0])
        fail(f"Column {j} is the first that differs from GPTQ's: got {np.round(Qa[:, j], 4).tolist()}, expected "
             f"{np.round(R[:, j], 4).tolist()}. Check how the error from column {j - 1 if j else 0} is passed on.")


@test("script_runs", "The script runs and prints RTN and GPTQ")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "GPTQ" not in stdout():
        fail("Nothing printed for GPTQ. Put the comparison loop back at the bottom.")
