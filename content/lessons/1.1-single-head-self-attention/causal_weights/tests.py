import numpy as np
import torch
from harness import test, fail, learner_fn, arr, LearnerError

rng = np.random.default_rng(7)
Q5, K5 = rng.standard_normal((2, 5, 4)).astype(np.float32)        # like the demo: T=5, d=4
Q16, K16 = rng.standard_normal((2, 5, 16)).astype(np.float32)     # d=16, where the 1/sqrt(d) matters
QB, KB = rng.standard_normal((2, 2, 3, 8)).astype(np.float32)     # a batch of 2, T=3, d=8


def softmax(x, axis=-1):
    m = np.max(np.where(np.isfinite(x), x, -np.inf), axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def scores(q, k, scale):
    return (q.astype(np.float64) @ np.swapaxes(k, -1, -2).astype(np.float64)) * scale


def causal(s, fill=-np.inf):
    T = s.shape[-1]
    return np.where(np.tril(np.ones((T, T))) == 1, s, fill)


def reference(q, k):
    return softmax(causal(scores(q, k, q.shape[-1] ** -0.5)))


def near(a, b):
    return a.shape == b.shape and np.allclose(a, b, rtol=1e-4, atol=1e-5)


def shape(a):
    return "[" + ", ".join(str(d) for d in a.shape) + "]"


def call(L, q, k):
    """Run the learner's causal_weights and return a NumPy array, explaining common crashes."""
    fn = learner_fn(L, "causal_weights")
    T, d = q.shape[-2], q.shape[-1]
    try:
        out = fn(torch.tensor(q), torch.tensor(k))
    except LearnerError as e:
        msg = str(e)
        if f"cannot be multiplied ({T}x{d} and {T}x{d})" in msg:
            fail(f"`q @ k` tries to multiply [{T}, {d}] by [{T}, {d}], and those don't fit. Every query needs a dot product "
                 "with every key, so the keys have to stand as columns. Which method swaps k's last two dimensions?")
        if q.ndim == 2 and f"tensor a ({d}) must match the size of tensor b ({T})" in msg:
            fail(f"Your scores come out [{d}, {d}], one per pair of *features*, so the [{T}, {T}] mask doesn't fit them. "
                 f"`q.transpose(-2, -1) @ k` sums over the tokens. You want one score per (query, key) pair: [{T}, {T}].")
        raise
    if out is None:
        fail("`causal_weights` returns nothing. It has to end with `return wei`.")
    if out is Ellipsis:
        fail("`causal_weights` returns `...`: `wei` is still a placeholder. Start with `scores` and work down.")
    a = arr(out)
    if a is None:
        fail(f"`causal_weights` should return a tensor, but it returns {type(out).__name__}.")
    return a.astype(np.float64)


@test("weights_shape", "Returns one weight per (query, key) pair: [T, T]")
def _(L):
    w = call(L, Q5, K5)
    if w.shape == (4, 4):
        fail("Your weights are [4, 4], one per pair of *features*. You want one per pair of *tokens*: [5, 5], "
             "query t against key s.")
    if w.shape != (5, 5):
        fail(f"For q and k of shape [5, 4] your weights have shape {shape(w)}. They should be [5, 5]: one row per query, one column per key.")


@test("causal", "No position attends to a later one")
def _(L):
    w = call(L, Q5, K5)
    upper = np.triu(np.ones((5, 5)), 1) == 1
    if np.isnan(w).any():
        row = int(np.argwhere(np.isnan(w))[0][0])
        if near(np.nan_to_num(w, nan=0.0), np.nan_to_num(softmax(np.where(np.tril(np.ones((5, 5))) == 1, -np.inf, scores(Q5, K5, 0.5))), nan=0.0)):
            fail(f"Row {row} of your weights is NaN. Your mask hides each token's past and keeps its future, so the last row "
                 "has nothing left, and softmax divides 0 by 0. Which cells should be hidden, where `tril` is 1 or where it's 0?")
        fail(f"Row {row} of your weights is NaN: every score in it was hidden, so softmax divided 0 by 0. Check which cells your mask hides.")
    if np.isneginf(w).any():
        finite = np.isfinite(w)
        if np.allclose(w[finite], scores(Q5, K5, 0.5)[finite], rtol=1e-4, atol=1e-5) or np.allclose(w[finite], scores(Q5, K5, 1.0)[finite], rtol=1e-4, atol=1e-5):
            fail("Your weights still hold −inf and raw scores. softmax is missing: it turns each row into positive weights that sum to 1, and −inf into exactly 0.")
        fail("There are −inf values in your weights, so the mask went on after softmax. Mask the scores first. Then softmax turns −inf into 0.")
    if np.abs(w[upper]).max() < 1e-6:
        return
    lower = np.tril(np.ones((5, 5)), -1) == 1
    if np.abs(w[lower]).max() < 1e-6:
        fail("Your weights are zero *below* the diagonal, so each token sees its future and not its past. "
             "`torch.triu` keeps the upper triangle. Which function keeps the lower one?")
    i, j = np.unravel_index(np.argmax(np.where(upper, np.abs(w), -1)), w.shape)
    leak = f"Token {i} puts {w[i, j]:.2f} of its weight on token {j}, which comes after it."
    for s in (0.5, 1.0):
        if near(w, softmax(causal(scores(Q5, K5, s), 0.0))):
            fail(f"{leak} You filled the future with 0, but softmax turns a score of 0 into e⁰ = 1, a real share of the weight. "
                 "Which score does softmax turn into exactly 0?")
        if near(w, softmax(scores(Q5, K5, s))):
            fail(f"{leak} Nothing is hidden yet. Which cells of the score grid are the future, and what value makes softmax give them 0?")
    fail(f"{leak} Hide every score above the diagonal before the softmax.")


@test("rows_sum_to_one", "Each row of weights sums to 1")
def _(L):
    w = call(L, Q5, K5)
    sums = w.sum(axis=-1)
    if np.allclose(sums, 1, atol=1e-5):
        return
    if np.allclose(w.sum(axis=0), 1, atol=1e-5):
        fail("Your *columns* sum to 1, not your rows. Row t holds query t's weights over the keys, so normalise along the last dimension.")
    if (sums < 1 - 1e-5).all() or np.isclose(sums[-1], 1, atol=1e-5):
        fail(f"Row 0 sums to {sums[0]:.2f}: you masked after the softmax, so the weight that went to the future is simply gone. "
             "Mask the scores first, then let softmax share 100% among the keys each token may see.")
    fail(f"Your rows sum to {', '.join(f'{v:.2f}' for v in sums)}. softmax along the last dimension makes each row sum to 1.")


@test("matches_reference", "Each row is softmax(q·k / √d) over the keys it may see")
def _(L):
    w = call(L, Q16, K16)
    ref = reference(Q16, K16)
    if near(w, ref):
        return
    peak = lambda a: a[1:].max(axis=-1).mean()
    stories = (
        (1.0, f"Your scores aren't scaled. With d = 16 features, dot products grow like √16 = 4, and softmax turns big gaps into "
              f"nearly one-hot rows: your rows peak at {peak(w):.2f} on average, the scaled ones at {peak(ref):.2f}. Divide by √d."),
        (1 / 16, "You divided by d (16), not √d (4), so the scores shrink too far and every row flattens towards an even average."),
        (4.0, "You multiplied by √d instead of dividing, so the scores grow even bigger and the rows go one-hot."),
        (0.5, "You divide by 2, which is √d only when d = 4, as in the demo. Here d = 16. Use the `d` you're given."),
        (5 ** -0.5, "You divided by √T, the number of tokens. The dot products grow with the number of *features*, so divide by √d."),
    )
    for s, story in stories:
        if near(w, softmax(causal(scores(Q16, K16, s)))):
            fail(story)
    if near(w, softmax(causal(scores(K16, Q16, 0.25)))):
        fail("Your grid is transposed: row t scores key t against every query. Row t should be *query* t against every key, so `q` goes on the left.")
    i, j = np.unravel_index(np.argmax(np.abs(w - ref)), ref.shape)
    fail(f"Weight [{i}, {j}] is {w[i, j]:.4f}, but softmax(q·k / √d) gives {ref[i, j]:.4f}. Check how you build `scores`.")


@test("batched", "Works on a batch of sequences, for any length T")
def _(L):
    try:
        w = call(L, QB, KB)
    except LearnerError as e:
        msg = str(e)
        if "must match the size of tensor b (5)" in msg:
            fail("Your mask is always 5 × 5, the demo's length. On sequences of 3 tokens it doesn't fit. Build it from the input: `T` is `q.shape[-2]`.")
        try:
            call(L, QB[0], KB[0])
        except LearnerError:
            raise e from None
        fail("It works on one sequence but not on a batch of [2, 3, 8]. `.T` reverses *every* dimension, so k.T is [8, 3, 2]. "
             "Swap only the last two: `k.transpose(-2, -1)`.")
    ref = reference(QB, KB)
    if w.shape != ref.shape:
        fail(f"On a batch of shape [2, 3, 8] your weights have shape {shape(w)}. They should be [2, 3, 3]: one grid per sequence.")
    if not near(w, ref):
        fail("On a batch of 2 sequences with d = 8, your weights differ from softmax(q·k / √d). Make every step work on the last two dimensions.")
