import numpy as np
import torch
from harness import skip, test, fail, learner_fn, arr

# 6 examples × 3 features: not square, and each feature on its own scale.
H = [[1.0, 10.0, -3.0],
     [2.0, 14.0, -1.0],
     [0.5, 6.0, -2.0],
     [3.0, 12.0, 0.0],
     [1.5, 8.0, -4.0],
     [2.5, 16.0, -2.5]]


def run(L, h, gamma=None, beta=None):
    h = torch.tensor(h)
    c = h.shape[1]
    gamma = torch.ones(c) if gamma is None else torch.tensor(gamma)
    beta = torch.zeros(c) if beta is None else torch.tensor(beta)
    out = learner_fn(L, "batchnorm")(h, gamma, beta)
    if out is Ellipsis:
        fail("`batchnorm` still returns `...`, the placeholder.")
    a = arr(out)
    if a is None:
        fail(f"`batchnorm` should return a tensor, but it returns {type(out).__name__}.")
    if a.shape != tuple(h.shape):
        fail(f"`batchnorm` returns shape {list(a.shape)} for an input of shape {list(h.shape)}. It should keep the input's shape.")
    return a.astype(np.float64)


def reference(h, gamma=None, beta=None, eps=1e-5):
    x = np.array(h, dtype=np.float64)
    gm = np.ones(x.shape[1]) if gamma is None else np.array(gamma)
    bt = np.zeros(x.shape[1]) if beta is None else np.array(beta)
    return gm * (x - x.mean(0)) / np.sqrt(x.var(0) + eps) + bt


def bessel(n):
    return (f"`.var()` and `.std()` divide by n − 1 by default (Bessel's correction, an unbiased estimate for a population). "
            f"In training, BatchNorm normalizes with the batch's own variance, dividing by n = {n}. Which argument switches the correction off?")


def basics_ok(L):
    """True when gamma = 1, beta = 0 already gives the reference answer."""
    try:
        return np.allclose(run(L, H), reference(H), atol=1e-3)
    except Exception:  # noqa: BLE001 - includes the test failures run() raises
        return False


@test("per_column", "Each feature gets mean 0 and std 1 over the batch")
def _(L):
    a = run(L, H)
    x = np.array(H, dtype=np.float64)
    if np.allclose(a.mean(0), 0, atol=1e-4) and np.allclose(a.std(0), 1, atol=1e-2):
        return
    if np.allclose(a, 1, atol=1e-4):
        fail("With gamma = 1 and beta = 0, every value comes out 1. gamma and beta are swapped: gamma multiplies the normalized value and beta is added.")
    if np.allclose(a.mean(1), 0, atol=1e-4):
        fail("Each *row* comes out with mean 0: you normalized every example across its own features. That's LayerNorm, which you'll meet in 1.2. "
             "BatchNorm normalizes each feature (column) across the batch. Which dim runs down the batch?")
    if abs(a.mean()) < 1e-4 and not np.allclose(a.mean(0), 0, atol=1e-4):
        fail("The whole batch has mean 0, but not each column. You used one mean and variance for everything, so a feature measured in "
             "tens still dwarfs one measured in ones. Each column needs its own statistics.")
    if not np.allclose(a.mean(0), 0, atol=1e-4):
        if np.allclose(a, x / np.sqrt(x.var(0) + 1e-5), atol=1e-3):
            fail("Each column is scaled but not centred: its mean isn't 0. Subtract the mean before dividing.")
        fail(f"Column means come out {np.round(a.mean(0), 3).tolist()}. After normalizing, each should be 0.")
    stds = a.std(0)
    var = x.var(0)
    n = len(H)
    if np.allclose(stds, np.sqrt((n - 1) / n), atol=1e-3):
        fail(f"Column stds come out {stds[0]:.3f}, a factor √({n - 1}/{n}) short of 1. " + bessel(n))
    if np.allclose(a, (x - x.mean(0)) / (var + 1e-5), atol=1e-3):
        fail(f"Column stds come out {np.round(stds, 3).tolist()}, not 1. You divided by the variance. "
             "Variance is in squared units; which statistic is in the same units as h?")
    fail(f"Column stds come out {np.round(stds, 3).tolist()}. After normalizing, each should be 1.")


@test("biased_var", "Matches nn.BatchNorm1d in training mode")
def _(L):
    if not basics_ok(L):
        skip("Fix the test above first.")
    small = H[:4]
    a = run(L, small)
    bn = torch.nn.BatchNorm1d(3)
    want = arr(bn(torch.tensor(small)).detach()).astype(np.float64)
    if np.allclose(a, want, atol=1e-4):
        return
    n = len(small)
    if np.allclose(a, want * np.sqrt((n - 1) / n), atol=1e-4):
        fail(f"Close, but every value is too small by a factor √({n - 1}/{n}). " + bessel(n))
    fail(f"On a batch of {n}, your output differs from nn.BatchNorm1d by up to {np.abs(a - want).max():.4f}.")


@test("eps_guard", "A constant feature doesn't turn into NaN")
def _(L):
    h = [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]
    if not basics_ok(L):
        skip("Fix the tests above first.")
    a = run(L, h)
    if np.isnan(a).any() or np.isinf(a).any():
        fail("A feature that's the same for every example has variance 0, and dividing 0 by √0 gives NaN. "
             "The `eps` argument exists for exactly this. Where does it go?")
    if not np.allclose(a, reference(h), atol=1e-3):
        fail("On a batch where the second feature is constant, your output doesn't match. That column should come out as all zeros.")


@test("gamma_beta", "gamma scales and beta shifts each feature")
def _(L):
    gamma, beta = [2.0, 0.5, 1.5], [1.0, -3.0, 0.25]
    if not basics_ok(L):
        skip("Fix the tests above first.")
    a = run(L, H, gamma, beta)
    want = reference(H, gamma, beta)
    if np.allclose(a, want, atol=1e-3):
        return
    if np.allclose(a, reference(H), atol=1e-3):
        fail(f"With gamma = {gamma} and beta = {beta}, every column still has mean 0 and std 1. "
             "The network learns gamma and beta so it can undo the normalization where that helps. Apply them after normalizing.")
    if np.allclose(a, reference(H, beta, gamma), atol=1e-3):
        fail("gamma and beta are swapped. gamma multiplies (it sets each column's std) and beta adds (it sets the mean).")
    fail(f"With gamma = {gamma} and beta = {beta}, your output is off by up to {np.abs(a - want).max():.3f}.")
