import re

import numpy as np
import torch
from harness import test, fail, warn, learner_fn, arr, script_error, stdout, LearnerError

# The lesson's tensors. Feature 2 is around −20 in every token; feature 5 spikes only in tokens 1 and 3.
X = [[0.42, -0.61, -18.5, 0.27, 0.95, 0.33],
     [-0.88, 0.14, -21.0, -0.52, 0.31, 9.6],
     [0.35, 0.72, -19.2, 0.60, -0.44, 0.21],
     [-0.27, -0.95, -17.8, 0.18, 0.66, 8.7]]
W = [[0.21, -0.43, 0.05, 0.37, -0.12, 0.08],
     [-0.30, 0.18, -0.07, 0.25, 0.41, -0.06],
     [0.15, 0.33, 0.04, -0.28, 0.19, 0.11]]


def f32(t):
    return np.array(t, dtype=np.float32)


def quantize_rows(t):
    s = (np.abs(t).max(axis=1, keepdims=True) / np.float32(127)).astype(np.float32)
    return np.clip(np.round(t / s), -127, 127).astype(np.int64), s.astype(np.float64)


def int8_ref(x, w):
    if x.shape[1] == 0:
        return np.zeros((x.shape[0], w.shape[0]))
    qx, sx = quantize_rows(x)
    qw, sw = quantize_rows(w)
    return (qx @ qw.T) * sx * sw.T


def split(mask, fp_part="float", int_part="int8"):
    """The product with the masked columns done one way and the rest another."""
    x, w = f32(X), f32(W)
    m = np.array(mask, dtype=bool)
    ways = {"float": lambda a, b: a.astype(np.float64) @ b.astype(np.float64).T, "int8": int8_ref, None: lambda a, b: 0}
    return ways[fp_part](x[:, m], w[:, m]) + ways[int_part](x[:, ~m], w[:, ~m])


def check_step1(L):
    fn = L.get("int8_matmul")
    if not callable(fn):
        return
    try:
        ok = np.allclose(arr(fn(torch.tensor(X), torch.tensor(W))), int8_ref(f32(X), f32(W)), rtol=1e-4, atol=1e-6)
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: int8_matmul no longer passes step 1's tests")


def call(L, threshold=None):
    fn = learner_fn(L, "mixed_matmul")
    args = (torch.tensor(X), torch.tensor(W)) + (() if threshold is None else (threshold,))
    try:
        out = fn(*args)
    except LearnerError as e:
        msg = str(e)
        m = re.search(r"size of axis is (\d+) but size of corresponding boolean axis is (\d+)", msg) or \
            re.search(r"shape of the mask \[(\d+)\].*indexed tensor \[(\d+)", msg)
        if m:
            sizes = {int(m.group(1)), int(m.group(2))}
            if sizes == {6, 4}:
                fail("Your mask has 4 flags, one per token, but it picks feature *columns*, and there are 6. "
                     "Reducing with `dim=1` asks \"does this token have a big value?\". Reduce over the tokens instead, `dim=0`.")
            if sizes == {6, 3}:
                fail("`W[big]` picks W's *rows*, and W has 3 of them, one per output channel. W is [out, features], "
                     "so its features are columns, like X's: `W[:, big]`.")
        raise
    if out is None:
        fail("`mixed_matmul` returns nothing. It has to return the sum of the two partial products.")
    if out is Ellipsis:
        fail("`mixed_matmul` still returns `...`, the placeholder. Start with `big`, the mask.")
    oa = arr(out)
    if oa is None:
        fail(f"`mixed_matmul` should return a tensor, but it returns {type(out).__name__}.")
    if oa.shape != (4, 3):
        fail(f"The result has shape [{', '.join(map(str, oa.shape))}], but X @ W.T is [4, 3].")
    return oa.astype(np.float64)


OUT = [False, False, True, False, False, True]


@test("splits_columns", "Outlier columns in full precision, the rest in int8")
def _(L):
    check_step1(L)
    out = call(L)
    near = lambda v: np.allclose(out, v, rtol=1e-4, atol=1e-6)
    if near(split(OUT)):
        return
    if near(split([False] * 5 + [True])):
        fail("Your split catches feature 5 but not feature 2, whose values are all around −20. "
             "`X > threshold` only sees big *positive* values. What makes −20 count as big?")
    if near(split([False, False, True, False, False, False])):
        fail("Your split catches feature 2 but misses feature 5. Feature 5 spikes only in tokens 1 and 3 (9.6 and 8.7), "
             "and `all` needs *every* token over the threshold. One token with a big value is enough to wreck that token's scale.")
    if near(split(OUT, fp_part=None)):
        fail("The outlier columns' contribution is never added back. A matmul sums over every feature, so dropping columns 2 and 5 "
             "changes the answer. Multiply them in full precision and add that part.")
    if near(split(OUT, fp_part="int8")):
        fail("The outlier columns went through int8 too, so all you've done is give them their own int8 scales. "
             "The point of LLM.int8() is that their part runs in full precision: plain `@` on the float columns.")
    if near(split(OUT, int_part=None)):
        fail("Only the outlier columns made it into the result. Add the int8 product of all the other columns.")
    if near(split([False] * 6)) or near(int8_ref(f32(X), f32(W))):
        fail("That's the same as plain int8 on everything. Pull the outlier columns out of the int8 path.")
    if near(split([not b for b in OUT])):
        fail("The two halves are swapped: the outlier columns went through int8 and the regular ones through full precision. "
             "`~big` should go to `int8_matmul`.")
    i, j = np.unravel_index(int(np.argmax(np.abs(out - split(OUT)))), out.shape)
    fail(f"Output [{i}, {j}] is {out[i, j]:.4f}, but the split product gives {split(OUT)[i, j]:.4f}. "
         "Check which columns each half gets.")


@test("uses_threshold", "The threshold argument decides what counts as an outlier")
def _(L):
    out = call(L, 20.0)
    only2 = [False, False, True, False, False, False]
    if np.allclose(out, split(only2), rtol=1e-4, atol=1e-6):
        return
    if np.allclose(out, split(OUT), rtol=1e-4, atol=1e-6):
        fail("With threshold=20.0, only feature 2 (−21.0 in token 1) should count as an outlier, but your split still "
             "pulls out feature 5, whose biggest value is 9.6. Use the `threshold` argument instead of a fixed number.")
    fail("With threshold=20.0, only feature 2 should go to full precision. Check how `big` uses `threshold`.")


@test("script_runs", "The script runs and prints both errors")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "mixed" not in stdout():
        fail("Nothing printed for the mixed matmul. Put the print line back so you can compare it with plain int8.")
