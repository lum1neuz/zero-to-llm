import numpy as np
import torch
from harness import test, fail, learner_fn, arr


def lesson_w():
    return 0.05 * torch.randn(256, generator=torch.Generator().manual_seed(0))


def other_w():
    """Two super-blocks, shifted off zero so the ranges are lopsided: hard-coded sizes can't pass."""
    return 0.03 * torch.randn(512, generator=torch.Generator().manual_seed(3)) + 0.01


def call(L, w):
    fn = learner_fn(L, "quantize_subblocks")
    out = fn(w)
    if out is None:
        fail("`quantize_subblocks` returns nothing. Keep the `return q, scale, lo` line.")
    if not isinstance(out, tuple) or len(out) != 3:
        fail("`quantize_subblocks` should return three things: `return q, scale, lo`.")
    q, scale, lo = out
    if q is Ellipsis and scale is Ellipsis:
        fail("`scale` and `q` are still `...`, the placeholders. Start with `scale`.")
    if q is Ellipsis:
        fail("`q` is still `...`, the placeholder. Build it from `wb`, `lo` and `scale`.")
    if scale is Ellipsis:
        fail("`scale` is still `...`, the placeholder.")
    qa, sa, la = arr(q), arr(scale), arr(lo)
    if qa is None or sa is None or la is None:
        fail("`q`, `scale` and `lo` should all be tensors.")
    return qa, sa.astype(np.float64), la.astype(np.float64)


def blocks(w):
    return arr(w).reshape(-1, 32).astype(np.float64)


@test("shapes", "One uint8 code per weight, one scale per sub-block")
def _(L):
    qa, sa, _la = call(L, lesson_w())
    if qa.dtype != np.uint8:
        if qa.dtype == np.int8:
            fail("`q` is int8. Codes count up from 0 and are never negative, so they're stored without a sign: `torch.uint8`.")
        fail(f"`q` is {qa.dtype.name}, more than one byte per code. Which dtype stores one byte, with no sign?")
    if qa.shape != (8, 32):
        fail(f"`q` has shape [{', '.join(map(str, qa.shape))}], but 256 weights in sub-blocks of 32 give [8, 32].")
    if sa.shape != (8, 1):
        fail(f"`scale` has shape [{', '.join(map(str, sa.shape))}]. Each of the 8 sub-blocks needs its own, as a [8, 1] column.")


@test("codes_fit", "Every code fits in 4 bits, 0…15")
def _(L):
    for w in (lesson_w(), other_w()):
        qa, _sa, _la = call(L, w)
        top = int(qa.max())
        if top > 200:
            fail(f"Some codes are around {top}: a negative code wrapped around in uint8, which has no negative numbers. "
                 "Q4_K's codes count up from each sub-block's lowest value, lo, so the lowest weight gets code 0 and nothing goes below it.")
        if top > 15:
            fail(f"The highest code is {top}, which doesn't fit in 4 bits (0…15). Check where `hi` lands with your scale, "
                 "and keep every code inside 0…15.")


@test("round_trip", "Every weight decodes to within scale / 2")
def _(L):
    for w in (lesson_w(), other_w()):
        qa, sa, la = call(L, w)
        wb = blocks(w)
        q = qa.astype(np.float64)
        err = np.abs(wb - (q * sa + la))
        if (err <= sa / 2 + 1e-7).all():
            continue
        lo, hi = wb.min(axis=1, keepdims=True), wb.max(axis=1, keepdims=True)
        near = lambda a, b: np.allclose(a, b, rtol=1e-4)
        if near(sa, (hi - lo) / 16):
            fail("16 codes have 15 steps between them, not 16. With `/ 16`, each step is too small, so `hi` needs code 16 "
                 "and gets cut back to 15, a whole step short. Which divisor makes code 15 land exactly on `hi`?")
        if near(sa, hi / 15):
            fail("Your scale only covers 0…hi, but the weights start at lo, below zero. The 16 codes should stretch across "
                 "the sub-block's whole range, from lo to hi.")
        # Recompute the learner's ratios in float32, as their code did, so floor and round agree to the bit.
        w32, s32, l32 = arr(w).reshape(-1, 32), sa.astype(np.float32), la.astype(np.float32)
        ratio = (w32 - l32) / s32
        if np.array_equal(q, np.clip(np.round(w32 / s32), 0, 15)):
            fail("Your codes are counted from zero, so every weight below zero lands on code 0. Code 0 stands for `lo` "
                 "(the way back adds `lo`), so count the steps up from `lo`.")
        if np.array_equal(q, np.clip(np.floor(ratio), 0, 15)) or np.array_equal(q, np.clip(np.trunc(ratio), 0, 15)):
            fail("Casting straight to uint8 chops off the decimals, so a weight 0.9 steps above a code still gets that code. "
                 "Which rounding goes to the *nearest* code?")
        k = np.unravel_index(int(np.argmax(err - sa / 2)), err.shape)
        fail(f"Weight {k[1]} of sub-block {k[0]} decodes {err[k]:.4f} away from its value, more than half a step ({sa[k[0], 0] / 2:.4f}). "
             "Check how `scale` and `q` come from `lo` and `hi`.")
