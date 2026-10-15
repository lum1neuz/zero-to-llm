import numpy as np
import torch
from harness import test, fail, learner_fn, arr, script_error, stdout, LearnerError

SALIENT = [1.0, 0.8, 6.0, 1.2, 0.9, 1.0, 3.0, 0.7]


def lesson_data():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(32, 8, generator=g) * torch.tensor(SALIENT)
    W = 0.3 * torch.randn(4, 8, generator=g)
    return X, W


def other_data():
    """16 tokens, other weights, salient channels elsewhere: so hard-coded data can't pass."""
    g = torch.Generator().manual_seed(21)
    X = torch.randn(16, 8, generator=g) * torch.tensor([4.0, 1.0, 1.0, 0.6, 1.0, 2.5, 1.0, 0.9])
    W = 0.3 * torch.randn(4, 8, generator=g)
    return X, W


def q_rows(w):
    s = np.abs(w).max(axis=1, keepdims=True) / np.float32(7)
    return np.clip(np.round(w / s), -7, 7) * s


def ref(X, W, alpha, stat=lambda x: np.abs(x).mean(axis=0), unscale=True, invert=False, src=None):
    x, w = arr(X), arr(W)
    s = stat(x if src is None else src(x, w)).astype(np.float32) ** np.float32(alpha)
    if invert:
        return q_rows(w / s) * s
    return q_rows(w * s) / s if unscale else q_rows(w * s)


def call(L, X, W, alpha):
    awq = learner_fn(L, "awq")
    try:
        out = awq(W, X, alpha)
    except LearnerError as e:
        if "must match the size of tensor" in str(e):
            fail(f"`s` has the wrong shape for `W * s`. It needs one factor per input channel, 8 of them, so it lines up with "
                 f"W's columns. Averaging over `dim=1` gives one per token ({X.shape[0]}) instead. Average over the tokens, `dim=0`.")
        raise
    if out is None:
        fail("`awq` returns nothing. It has to return the quantized weights, scaled back.")
    if out is Ellipsis:
        fail("`awq` still returns `...`, the placeholder. Start with `s`.")
    oa = arr(out)
    if oa is None:
        fail(f"`awq` should return a tensor, but it returns {type(out).__name__}.")
    return oa


@test("shape", "awq returns W_hat with W's shape")
def _(L):
    X, W = lesson_data()
    oa = call(L, X, W, 0.5)
    if oa.shape != (4, 8):
        fail(f"`awq` returns shape [{', '.join(map(str, oa.shape))}], but W is [4, 8]. W_hat replaces W one for one.")


@test("activation_aware", "Columns are scaled by mean|x|^α before quantizing, and scaled back after")
def _(L):
    for X, W in (lesson_data(), other_data()):
        for alpha in (0.5, 1.0, 0.0):
            oa = call(L, X, W, alpha)
            near = lambda v: np.allclose(oa, v, rtol=1e-4, atol=1e-6)
            if near(ref(X, W, alpha)):
                continue
            if np.isnan(oa).any():
                col = int(np.flatnonzero(arr(X).mean(axis=0) < 0)[0])
                fail(f"W_hat is nan. Without `.abs()`, channel {col}'s activations average to a negative number, "
                     "and a negative number to a fractional power is nan. Average the *magnitudes*.")
            if near(ref(X, W, alpha, unscale=False)):
                fail("You quantized `W * s` but never divided back by s, so every salient column comes out s times too big. "
                     "X still multiplies the original W, so W_hat has to be in W's own units.")
            if near(ref(X, W, alpha, invert=True)):
                fail("You shrank the salient columns before quantizing (W / s), so the weights that matter most got the *coarsest* steps. "
                     "Stretch them instead, then undo it.")
            if near(ref(X, W, alpha, src=lambda x, w: w)):
                fail("Your factors come from the weights, `W.abs().mean(dim=0)`. AWQ is activation-aware: a column matters because "
                     "its channel's *activations* are big. Measure X.")
            if near(ref(X, W, alpha, stat=lambda x: np.abs(x).max(axis=0))):
                fail("Your factors use the largest |x| per channel. One spiky token then decides the factor. "
                     "AWQ uses the average |x| over the calibration tokens, which says how much the channel matters typically.")
            if alpha != 0.5 and near(ref(X, W, 0.5)):
                fail(f"With alpha={alpha}, your W_hat is the same as with 0.5. Use the `alpha` argument instead of a fixed number.")
            if alpha == 0.0 and not near(q_rows(arr(W))):
                fail("With alpha=0 every factor is 1, so W_hat should be plain round-to-nearest. Yours isn't. Check what multiplies W.")
            i, j = np.unravel_index(int(np.argmax(np.abs(oa - ref(X, W, alpha)))), oa.shape)
            fail(f"With alpha={alpha}, W_hat[{i}, {j}] is {oa[i, j]:.4f}, but stretching column {j} by mean|x_{j}|^α, quantizing "
                 f"and shrinking it back gives {ref(X, W, alpha)[i, j]:.4f}.")


@test("script_runs", "The script runs and prints all five alphas")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if stdout().count("alpha") < 5:
        fail("The alpha loop didn't print all five lines. Put the loop back at the bottom.")
