import numpy as np
import torch
from harness import test, fail, learner_fn, arr, script_error, stdout, LearnerError

# The lesson's tensors. Feature 2 is around −20 in every token and all negative.
X = [[0.42, -0.61, -18.5, 0.27, 0.95, 0.33],
     [-0.88, 0.14, -21.0, -0.52, 0.31, 9.6],
     [0.35, 0.72, -19.2, 0.60, -0.44, 0.21],
     [-0.27, -0.95, -17.8, 0.18, 0.66, 8.7]]
W = [[0.21, -0.43, 0.05, 0.37, -0.12, 0.08],
     [-0.30, 0.18, -0.07, 0.25, 0.41, -0.06],
     [0.15, 0.33, 0.04, -0.28, 0.19, 0.11]]

x64 = np.array(X, dtype=np.float32).astype(np.float64)
w64 = np.array(W, dtype=np.float32).astype(np.float64)


def num(v):
    return f"{v:.1f}".replace("-", "−")


def ref_s(alpha):
    return np.abs(x64).max(axis=0) ** alpha / np.abs(w64).max(axis=0) ** (1 - alpha)


def call(L, alpha=None):
    fn = learner_fn(L, "smooth")
    args = (torch.tensor(X), torch.tensor(W)) + (() if alpha is None else (alpha,))
    try:
        out = fn(*args)
    except LearnerError as e:
        msg = str(e)
        if "must match the size of tensor" in msg:
            fail(f"One of the two maxes runs along the wrong dimension: {msg.split('RuntimeError: ')[-1]}. "
                 "Both need one value per feature, 6 of them. X is [tokens, features] and W is [out, features], "
                 "so the features are the columns of both: reduce over dim 0 for both.")
        raise
    if out is None:
        fail("`smooth` returns nothing. It has to return the two rescaled tensors: `return X_new, W_new`.")
    if out is Ellipsis:
        fail("`smooth` still returns `...`, the placeholder. Start with `s`, one factor per feature.")
    if not isinstance(out, tuple) or len(out) != 2:
        got = "one value" if not isinstance(out, tuple) else f"{len(out)} values"
        fail(f"`smooth` returns {got}. It has to return two, the smoothed X and the smoothed W.")
    xs, ws = (arr(t) for t in out)
    if xs is None or ws is None:
        fail("`smooth` should return two tensors.")
    return xs.astype(np.float64), ws.astype(np.float64)


@test("returns_pair", "Returns the smoothed X and W, in their original shapes")
def _(L):
    xs, ws = call(L)
    if xs.shape == (3, 6) and ws.shape == (4, 6):
        fail("The two tensors come back in the wrong order. Return the activations first, then the weights.")
    if xs.shape != (4, 6) or ws.shape != (3, 6):
        fail(f"`smooth` returns shapes [{', '.join(map(str, xs.shape))}] and [{', '.join(map(str, ws.shape))}]. "
             "X is [4, 6] and W is [3, 6], and smoothing doesn't change either.")


@test("same_product", "X_s @ W_s.T still equals X @ W.T")
def _(L):
    xs, ws = call(L)
    if np.isnan(xs).any() or np.isnan(ws).any():
        col = int(np.argwhere(np.isnan(xs).any(axis=0) | np.isnan(ws).any(axis=0))[0][0])
        fail(f"Feature {col} of your result is nan. Its values are all negative, so without `.abs()` its max is "
             f"{num(x64[:, col].max())}, and a negative number raised to a fractional power is nan. "
             "Take the max of the *magnitudes*.")
    y = x64 @ w64.T
    if np.allclose(xs @ ws.T, y, rtol=1e-4, atol=1e-5):
        return
    ratio_x = x64[0] / xs[0]   # what X was divided by, per feature
    ratio_w = ws[0] / w64[0]   # what W was multiplied by
    if np.allclose(ratio_x, 1 / ratio_w, rtol=1e-3):
        both = "divided both X and W by s, so X @ W.T shrinks" if ratio_x[2] > 1 else "multiplied both X and W by s, so X @ W.T grows"
        fail(f"You {both} by s² on every feature. "
             "For the product to stay the same, one side divides and the other multiplies.")
    fail("X_s @ W_s.T doesn't match X @ W.T anymore. Whatever you divide feature j of X by, multiply feature j of W by the same number.")


@test("migrates", "Each feature's factor is max|X_j|^α / max|W_j|^(1−α)")
def _(L):
    for alpha in (0.5, 0.75):
        xs, ws = call(L, alpha)
        s = x64[0] / xs[0]   # X has no zeros in row 0, so this recovers the learner's factors
        ref = ref_s(alpha)
        if np.allclose(s, ref, rtol=1e-3):
            continue
        if np.allclose(s, 1 / ref, rtol=1e-3):
            fail(f"Feature 2 of X got bigger: −18.5 became {num(xs[0, 2])}. You multiplied the activations by s and "
                 "divided the weights. Divide X by s so its outlier shrinks, and let W, which has room to spare, pick up the factor.")
        if alpha != 0.5 and np.allclose(s, ref_s(0.5), rtol=1e-3):
            fail("With alpha=0.75 your factors are the same as with 0.5. Use the `alpha` argument in both powers "
                 "instead of a fixed number.")
        if np.allclose(s, np.abs(x64).max(axis=0) ** (1 - alpha) / np.abs(w64).max(axis=0) ** alpha, rtol=1e-3):
            fail(f"With alpha={alpha}, the exponents are swapped: max|X| is raised to 1 − alpha and max|W| to alpha. "
                 "Alpha is how much of the activations' range moves out, so it goes on max|X|.")
        if np.allclose(s, np.abs(x64).mean(axis=0) ** alpha / np.abs(w64).mean(axis=0) ** (1 - alpha), rtol=1e-3):
            fail("Your factors use the mean of |x| and |w|. SmoothQuant balances the largest values, because the max is what "
                 "sets each side's int8 scale. Use the max over each feature.")
        j = int(np.argmax(np.abs(s - ref) / ref))
        fail(f"With alpha={alpha}, feature {j}'s factor is {s[j]:.4f}, but max|X_{j}|^α / max|W_{j}|^(1−α) is {ref[j]:.4f}.")


@test("script_runs", "The script runs and prints the smoothed error")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "smooth" not in stdout():
        fail("Nothing printed for the smoothed matmul. Put the print line back so you can compare it with the other two.")
