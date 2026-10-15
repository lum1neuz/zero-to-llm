import numpy as np
import torch
from harness import test, fail, learner_fn, arr


def lesson_X():
    """The lesson's calibration batch, drawn exactly as the script draws it. Feature 7 is dead."""
    g = torch.Generator().manual_seed(6)
    mix = torch.randn(3, 8, generator=g)
    X = torch.randn(64, 3, generator=g) @ mix + 0.1 * torch.randn(64, 8, generator=g)
    X[:, 7] = 0.0
    return X


def other_X():
    """32 tokens, 5 features, nothing dead: other shapes, so hard-coded sizes can't pass."""
    g = torch.Generator().manual_seed(11)
    return torch.randn(32, 5, generator=g) * torch.tensor([0.5, 2.0, 1.0, 3.0, 0.8])


def call(L, X, *rest):
    hessian = learner_fn(L, "hessian")
    H = hessian(X, *rest)
    if H is None:
        fail("`hessian` returns nothing. It has to end with `return` and the damped H.")
    if H is Ellipsis:
        fail("`hessian` still returns `...`, the placeholder. Start with `H`.")
    Ha = arr(H)
    if Ha is None:
        fail(f"`hessian` should return a tensor, but it returns {type(H).__name__}.")
    return Ha.astype(np.float64)


def fit(Ha, X):
    """Learner's H as c · XᵀX + D. c comes from the off-diagonal entries (allowing a constant offset, which is
    damping added to every entry and gets its own diagnosis)."""
    x = arr(X).astype(np.float64)
    H0 = x.T @ x
    off = ~np.eye(len(H0), dtype=bool)
    A = np.stack([H0[off], np.ones(off.sum())], axis=1)
    (c, _b), *_ = np.linalg.lstsq(A, Ha[off], rcond=None)
    return H0, float(c), Ha - c * H0


@test("feature_square", "H is [features, features] and follows XᵀX")
def _(L):
    for X in (lesson_X(), other_X()):
        n, d = X.shape
        Ha = call(L, X)
        if Ha.shape == (n, n):
            fail(f"H has shape [{n}, {n}], one row per calibration *token*: that's X @ X.T. GPTQ needs how each pair of "
                 f"*features* moves together, so H is [{d}, {d}]. Which side does the transpose go on?")
        if Ha.shape != (d, d):
            fail(f"H has shape [{', '.join(map(str, Ha.shape))}], but X has {d} features, so H should be [{d}, {d}].")
        H0, c, D = fit(Ha, X)
        off = ~np.eye(d, dtype=bool)
        if c <= 0 or np.ptp(D[off]) > 1e-4 * np.abs(c * H0).max():
            fail("H's off-diagonal entries don't follow XᵀX. Entry [i, j] should sum feature i times feature j over all the tokens.")


@test("damped", "Damping adds percdamp × the diagonal's mean, on the diagonal only")
def _(L):
    for X, pd in ((lesson_X(), 0.01), (lesson_X() * 10, 0.01), (other_X(), 0.1)):
        Ha = call(L, X, pd)
        d = X.shape[1]
        if Ha.shape != (d, d):
            fail(f"H has shape [{', '.join(map(str, Ha.shape))}] instead of [{d}, {d}]. Fix \"H is [features, features]\" above first.")
        H0, c, D = fit(Ha, X)
        off = ~np.eye(d, dtype=bool)
        tol = 1e-4 * np.abs(c * H0).max()
        diag = np.diag(D)
        mean_diag = c * np.diag(H0).mean()
        dead = np.flatnonzero(np.diag(H0) == 0)
        if np.abs(D).max() <= tol:
            if len(dead):
                fail(f"There's no damping. Feature {dead[0]} never fires in this calibration batch, so H[{dead[0]}, {dead[0]}] is 0 "
                     "and H has no inverse, which GPTQ's next step needs. Add a little to every diagonal entry: "
                     "`percdamp` (1%) of the diagonal's mean.")
            fail("There's no damping on H's diagonal. Add `percdamp` times the mean of the diagonal to every diagonal entry.")
        if np.abs(D[off]).max() > tol:
            fail("Your damping landed on every entry of H, off-diagonals too, which changes how features relate to each other. "
                 "Add it to the diagonal only, for example with `torch.eye`.")
        if np.ptp(diag) > tol:
            if len(dead) and abs(Ha[dead[0], dead[0]]) <= tol:
                fail(f"Each diagonal entry got a different amount, in proportion to itself, so H[{dead[0]}, {dead[0]}] stays 0 "
                     f"for feature {dead[0]}, which never fires. Add the *same* amount to every diagonal entry: "
                     "percdamp × the diagonal's mean.")
            fail("Each diagonal entry got a different amount. Add the same number to all of them: percdamp × the diagonal's mean.")
        got = float(diag.mean())
        if np.isclose(got, pd * mean_diag, rtol=1e-3):
            continue
        if np.isclose(got, pd, rtol=1e-3):
            fail(f"Your damping is {pd} whatever H's size. With X ten times bigger, H would be 100 times bigger and {pd} would barely "
                 f"registers. Make it relative: percdamp × the mean of H's diagonal.")
        if np.isclose(got, pd * c * H0.mean(), rtol=1e-3):
            fail("Your damping uses the mean of every entry of H, off-diagonal entries included. GPTQ scales it by the mean of the "
                 "*diagonal*, each feature's own energy. Pick the diagonal out with `H[idx, idx]`.")
        if pd != 0.01 and np.isclose(got, 0.01 * mean_diag, rtol=1e-3):
            fail(f"With percdamp={pd}, your damping is still 1% of the diagonal's mean. Use the `percdamp` argument "
                 "instead of a fixed 0.01.")
        fail(f"Your damping adds {got:.4g} to the diagonal, but percdamp × the diagonal's mean is {pd * mean_diag:.4g}.")
