import numpy as np
import torch
from harness import test, fail, learner_fn, arr, script_error, stdout, LearnerError

NF4 = np.array([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
                0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0], dtype=np.float32)


def lesson_w():
    return 0.05 * torch.randn(512, generator=torch.Generator().manual_seed(0))


def other_w():
    """96 weights in blocks of 32, another seed and spread: hard-coded sizes can't pass."""
    return 0.02 * torch.randn(96, generator=torch.Generator().manual_seed(5))


def ref_codes(w, block, normalize=True, dist=lambda d: np.abs(d), pick=np.argmin):
    wb = arr(w).reshape(-1, block)
    am = np.abs(wb).max(axis=1, keepdims=True)
    wn = wb / am if normalize else wb
    return pick(dist(wn[..., None] - NF4), axis=-1), am


def quantize(L, w, block):
    fn = learner_fn(L, "nf4_quantize")
    try:
        out = fn(w, block)
    except LearnerError as e:
        if "must match the size of tensor" in str(e):
            fail(f"`wn - NF4` tries to line the 16 levels up with the {block} weights in each block. Give `wn` a new last axis, "
                 f"`wn[..., None]`, so that every weight meets all 16 levels: a [blocks, {block}, 16] table of gaps.")
        raise
    if not isinstance(out, tuple) or len(out) != 2:
        fail("`nf4_quantize` should return two things, the codes and the absmax per block. Keep the `return idx.to(torch.uint8), absmax` line.")
    idx, am = arr(out[0]), arr(out[1])
    if idx is None or am is None:
        fail("`nf4_quantize` should return two tensors.")
    return idx, am


@test("codes", "One uint8 code per weight, 0 to 15")
def _(L):
    w = lesson_w()
    idx, _am = quantize(L, w, 64)
    if idx.shape == (8, 16):
        fail("`idx` has shape [8, 16]: the argmin ran over the block's 64 weights, so you got one answer per *level*. "
             "Reduce over the last axis, the 16 levels, to get one code per weight: [8, 64].")
    if idx.shape != (8, 64):
        fail(f"`idx` has shape [{', '.join(map(str, idx.shape))}], but 512 weights in blocks of 64 need [8, 64], one code per weight.")
    if idx.min() < 0 or idx.max() > 15:
        fail(f"Codes run from {idx.min()} to {idx.max()}, but there are only 16 levels, so a code is an index from 0 to 15.")


@test("nearest_level", "Each code points at the nearest NF4 level")
def _(L):
    for w, block in ((lesson_w(), 64), (other_w(), 32)):
        idx, _am = quantize(L, w, block)
        want, _ = ref_codes(w, block)
        if idx.shape != want.shape:
            fail(f"`idx` has shape [{', '.join(map(str, idx.shape))}] instead of one code per weight. Fix \"One uint8 code per weight\" above first.")
        if np.array_equal(idx, want):
            continue
        same = lambda v: idx.shape == v.shape and np.array_equal(idx, v)
        if same(ref_codes(w, block, normalize=False)[0]):
            fail("Each block was never divided by its block's absmax, so the weights, all within about ±0.15, only ever reach "
                 "the few levels near zero. Scale every block into [−1, 1] first.")
        if same(ref_codes(w, block, dist=lambda d: d)[0]):
            fail("Every weight got code 15. The argmin ran over signed gaps, weight minus level, and the most negative gap is "
                 "always the level furthest above. Turn the gaps into distances first.")
        if same(ref_codes(w, block, pick=np.argmax)[0]):
            fail("Your codes point at the level *furthest* from each weight. You want the closest one.")
        wb = arr(w).reshape(-1, block)
        wn = wb / np.abs(wb).max(axis=1, keepdims=True)
        if same(np.round(wn * 7) + 7) or same(np.round(wn * 7.5) + 7.5) or same(np.round((wn + 1) * 7.5)):
            fail("Your codes treat the 16 levels as if they were evenly spaced, like int4's. NF4's levels crowd together "
                 "near zero, so rounding lands on the wrong ones. Measure the distance to each level and pick the closest.")
        k = np.argwhere(idx != want)[0]
        fail(f"Weight {k[1]} of block {k[0]} normalizes to {wn[k[0], k[1]]:.4f}. Its nearest level is {NF4[want[tuple(k)]]:.4f} "
             f"(code {want[tuple(k)]}), but your code {idx[tuple(k)]} points at {NF4[idx[tuple(k)]]:.4f}.")


@test("round_trip", "nf4_dequantize looks the levels up, rescales and flattens")
def _(L):
    for w, block in ((lesson_w(), 64), (other_w(), 32)):
        idx, am = ref_codes(w, block)
        fn = learner_fn(L, "nf4_dequantize")
        try:
            out = fn(torch.tensor(idx.astype(np.uint8)), torch.tensor(am))
        except LearnerError as e:
            msg = str(e)
            if "too many indices" in msg or "mask" in msg:
                fail("`idx` is uint8, and torch reads a uint8 (or bool) index as a *mask*, not as positions. "
                     "Turn the codes into int64 positions first with `.long()`.")
            raise
        if out is Ellipsis:
            fail("`nf4_dequantize` still returns `...`, the placeholder.")
        oa = arr(out)
        if oa is None:
            fail(f"`nf4_dequantize` should return a tensor, but it returns {type(out).__name__}.")
        n = arr(w).size
        if oa.shape == idx.shape:
            fail(f"`w_hat` has shape [{', '.join(map(str, oa.shape))}], one row per block. Flatten it back into one row of {n} "
                 "so it lines up with `w`.")
        if oa.shape != (n,):
            fail(f"`w_hat` has shape [{', '.join(map(str, oa.shape))}], but `w` is [{n}].")
        want = (NF4[idx] * am).reshape(-1)
        if np.allclose(oa, want, rtol=1e-5, atol=1e-7):
            continue
        if np.allclose(oa, NF4[idx].reshape(-1)):
            fail("Every value of `w_hat` is between −1 and 1: those are the levels themselves. Each block was divided by its "
                 "absmax on the way in, so multiply by it on the way out.")
        i = int(np.argmax(np.abs(oa - want)))
        fail(f"`w_hat[{i}]` is {oa[i]:.5f}, but its level times its block's absmax is {want[i]:.5f}.")


@test("script_runs", "The script runs and prints both errors")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "NF4" not in stdout():
        fail("Nothing printed for NF4. Put the comparison loop back at the bottom.")
