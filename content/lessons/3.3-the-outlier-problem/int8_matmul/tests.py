import numpy as np
import torch
from harness import test, fail, learner_fn, arr, LearnerError

# The lesson's tensors: 4 tokens × 6 features with outlier features 2 and 5, and a 3 × 6 layer.
X = [[0.42, -0.61, -18.5, 0.27, 0.95, 0.33],
     [-0.88, 0.14, -21.0, -0.52, 0.31, 9.6],
     [0.35, 0.72, -19.2, 0.60, -0.44, 0.21],
     [-0.27, -0.95, -17.8, 0.18, 0.66, 8.7]]
W = [[0.21, -0.43, 0.05, 0.37, -0.12, 0.08],
     [-0.30, 0.18, -0.07, 0.25, 0.41, -0.06],
     [0.15, 0.33, 0.04, -0.28, 0.19, 0.11]]
# Other shapes (2 tokens, 5 outputs) and other values, so hard-coded shapes or the script's
# globals can't pass by accident.
X2 = [[0.5, -1.2, 0.3, 2.0, -0.7, 0.1],
      [1.1, 0.4, -0.9, -0.2, 0.6, -1.5]]
W2 = [[0.12, -0.05, 0.33, 0.08, -0.21, 0.17],
      [-0.44, 0.29, 0.02, -0.16, 0.37, 0.06],
      [0.09, 0.51, -0.27, 0.14, -0.03, -0.38],
      [0.25, -0.11, 0.19, -0.42, 0.07, 0.31],
      [-0.06, 0.22, -0.48, 0.03, 0.15, -0.09]]


def num(v):
    return str(v).replace("-", "−")


def shape(a):
    return "[" + ", ".join(str(d) for d in a.shape) + "]"


def quantize_rows(t):
    t = np.array(t, dtype=np.float32)
    s = (np.abs(t).max(axis=1, keepdims=True) / np.float32(127)).astype(np.float32)
    return np.clip(np.round(t / s), -127, 127).astype(np.int64), s.astype(np.float64)


def parts(x, w):
    qx, sx = quantize_rows(x)
    qw, sw = quantize_rows(w)
    acc = qx @ qw.T
    return acc, sx, sw


def wrap8(a):
    return (a + 128) % 256 - 128


def call(L, x, w):
    fn = learner_fn(L, "int8_matmul")
    try:
        out = fn(torch.tensor(x), torch.tensor(w))
    except LearnerError as e:
        msg = str(e)
        if "must match the size of tensor" in msg:
            n, m = len(x), len(w)
            fail(f"Scaling `acc` back failed: {msg.split('RuntimeError: ')[-1]}. "
                 f"`acc` is [{n}, {m}]. `sx` is a [{n}, 1] column, one scale per token, which lines up with acc's rows. "
                 f"`sw` is a [{m}, 1] column too, one per output channel, but output channels run along acc's *columns*. "
                 "Turn it into a row with `sw.T`.")
        raise
    if out is None:
        fail("`int8_matmul` returns nothing. It has to return the rescaled product.")
    if out is Ellipsis:
        fail("`int8_matmul` still returns `...`, the placeholder. Fill in `acc` first, then the return line.")
    oa = arr(out)
    if oa is None:
        fail(f"`int8_matmul` should return a tensor, but it returns {type(out).__name__}.")
    return oa


@test("returns_float", "Returns floats in X @ W.T's shape")
def _(L):
    out = call(L, X, W)
    if out.shape == (3, 4):
        fail("The result has shape [3, 4], one row per output channel. X @ W.T has one row per token: [4, 3]. Put `qx` on the left.")
    if out.shape != (4, 3):
        fail(f"The result has shape {shape(out)}, but X @ W.T is [4, 3]: one row per token, one column per output channel.")
    if out.dtype.kind != "f":
        fail(f"`int8_matmul` returns {out.dtype.name} integers. They're still in units of the two scales, "
             "so an output near 1.5 comes back in the thousands. Scale them back with `sx` and `sw`.")


@test("no_wrap", "Integer products add up without wrapping")
def _(L):
    out = call(L, X, W).astype(np.float64)
    acc, sx, sw = parts(X, W)
    wrapped = wrap8(acc)
    if np.allclose(out, wrapped * sx * sw.T, rtol=1e-4, atol=1e-6) and not np.array_equal(acc, wrapped):
        i, j = np.argwhere(acc != wrapped)[0]
        fail(f"Your integer products wrapped around. `qx @ qw.T` stays in int8, which only holds −128…127, "
             f"but token {i} against channel {j} sums to {num(acc[i, j])} and comes back as {num(wrapped[i, j])}. "
             "int8 hardware multiplies int8s but adds the products up in int32. Cast both sides with `.int()` before `@`.")


@test("rescaled", "Each output gets its token's scale times its channel's scale")
def _(L):
    for x, w in ((X, W), (X2, W2)):
        out = call(L, x, w).astype(np.float64)
        acc, sx, sw = parts(x, w)
        ref = acc * sx * sw.T
        if out.shape == ref.shape and np.allclose(out, ref, rtol=1e-4, atol=1e-6):
            continue
        exact = np.array(x, dtype=np.float32).astype(np.float64) @ np.array(w, dtype=np.float32).astype(np.float64).T
        near = lambda v: out.shape == v.shape and np.allclose(out, v, rtol=1e-4, atol=1e-6)
        if near(exact):
            fail("Your result is exactly X @ W.T, so nothing was quantized. Multiply the int8 versions, `qx` and `qw`, "
                 "and scale the result back.")
        if near(acc * sx):
            fail("Your result never uses `sw`, so every output is still in units of its channel's scale. "
                 "Output [i, j] came from token i *and* channel j. Multiply by both scales.")
        if near(acc * sw.T):
            fail("Your result never uses `sx`, so every output is still in units of its token's scale. "
                 "Output [i, j] came from token i *and* channel j. Multiply by both scales.")
        if near(acc / sx / sw.T):
            fail("You divided by the scales. Quantizing divided the floats by them, so going back multiplies.")
        i, j = np.unravel_index(int(np.argmax(np.abs(out - ref))), ref.shape)
        fail(f"Output [{i}, {j}] is {out[i, j]:.4f}, but the int8 product scaled back gives {ref[i, j]:.4f}. "
             "Multiply `acc` by token i's scale and channel j's scale.")
