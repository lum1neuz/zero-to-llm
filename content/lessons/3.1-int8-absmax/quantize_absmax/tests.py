import numpy as np
import torch
from harness import test, fail, learner_fn, arr, as_float

W = [0.42, -1.30, 0.07, 0.88, -0.25, 2.10, -0.61, 0.33]
OUTLIER = [-3.0, 1.0, 0.5, -0.2]
ROW = "`[-3.0, 1.0, 0.5, -0.2]`"


def num(v):
    """Format like the lesson text: one decimal for floats, a real minus sign."""
    text = f"{v:.1f}" if isinstance(v, float) else str(v)
    return text.replace("-", "−")


def run_on(L, values):
    """Call the learner's quantize_absmax and unpack it, explaining any shape mistakes."""
    quantize = learner_fn(L, "quantize_absmax")
    out = quantize(torch.tensor(values))
    if out is None:
        fail("`quantize_absmax` returns nothing. It has to end with `return q, scale`.")
    if not isinstance(out, tuple) or len(out) != 2:
        got = "one value" if not isinstance(out, tuple) else f"{len(out)} values"
        fail(f"`quantize_absmax` returns {got}. It has to return two: `return q, scale`.")
    q, scale = out
    if scale is Ellipsis and q is Ellipsis:
        fail("`scale` and `q` are still `...`, the placeholders. Start with `scale`.")
    if scale is Ellipsis:
        fail("`scale` is still `...`, the placeholder. Which number maps the largest weight onto 127?")
    if q is Ellipsis:
        fail("`q` is still `...`, the placeholder. Build it from `w` and `scale`.")
    s = as_float(scale)
    if s is None:
        fail(f"`scale` should be a single number (a 0-d tensor), but it's {type(scale).__name__}.")
    qa = arr(q)
    if qa is None:
        fail(f"`q` should be a tensor, but it's {type(q).__name__}.")
    return qa, s


def scale_story(values, s):
    """Name the specific mistake behind a wrong scale, from its value alone."""
    x = np.array(values, dtype=np.float32)
    amax = float(np.abs(x).max())
    biggest = float(x[np.abs(x).argmax()])
    lands = round(biggest / s) if s else float("inf")
    near = lambda a, b: abs(a - b) <= 1e-4 * max(1.0, abs(b))
    if near(s, amax / 127):
        return None
    if near(s, float(x.max()) / 127):
        return (f"On the row {ROW} your scale comes from {num(float(x.max()))}, not {num(biggest)}, "
                f"so {num(biggest)} would land on {num(lands)}. Which weight has the largest *magnitude*?")
    if near(s, 127 / amax):
        return "Your scale is 127 divided by the max. Flip it, so that `w / scale` turns the biggest weight into 127."
    if near(s, float(np.abs(x).mean()) / 127) or near(s, abs(float(x.mean())) / 127) or near(s, float(x.mean()) / 127):
        return f"The mean doesn't bound anything, so the biggest weight would land on {num(lands)}. Which statistic does?"
    if near(s, amax / 128):
        return "Dividing by 128 puts the biggest weight on ±128, and +128 doesn't fit in int8. What's the largest value int8 can hold?"
    if near(s, amax / 255) or near(s, amax / 256):
        return f"The biggest weight lands on {num(lands)}, outside int8's range of −128…127. Check the number you divide by."
    if near(s, amax):
        return "Your scale is the max itself, so every weight lands between −1 and 1. Divide by the int8 limit too."
    return f"With your scale, the biggest weight on {ROW} lands on {num(lands)} instead of ±127. Check how you compute `scale`."


@test("returns_int8", "Returns q and scale, with q stored as int8")
def _(L):
    qa, _s = run_on(L, W)
    if qa.dtype != np.int8:
        name = {"float32": "float32, which is 4 bytes per weight", "int64": "int64, which is 8 bytes per weight"}.get(qa.dtype.name, qa.dtype.name)
        fail(f"`q` is still {name}. Which dtype stores exactly one byte?")


@test("largest_is_127", "The largest |w| lands on ±127")
def _(L):
    qa, s = run_on(L, OUTLIER)
    story = scale_story(OUTLIER, s)
    if story:
        fail(story)


@test("in_range", "Every q fits in [−127, 127]")
def _(L):
    for values in (W, OUTLIER):
        qa, s = run_on(L, values)
        x = np.array(values, dtype=np.float32)
        q = qa.astype(np.int64)
        flipped = (np.sign(q) * np.sign(x) < 0) & (np.abs(x) >= 0.5 * np.abs(x).max())
        if qa.dtype == np.int8 and flipped.any():
            fail("A value went past 127 and wrapped around when cast to int8 (128 becomes −128). Keep `q` inside the range before you cast.")
        if (q < -127).any() or (q > 127).any():
            bad = int(q[(q < -127) | (q > 127)][0])
            fail(f"A value landed on {num(bad)}, outside [−127, 127]. Keeping the grid symmetric means −128 stays unused.")


@test("round_trip", "Round-trip error stays within scale / 2")
def _(L):
    qa, s = run_on(L, W)
    x = np.array(W, dtype=np.float32)
    q = qa.astype(np.float64)
    err = float(np.abs(x - q * s).max())
    if err <= s / 2 + 1e-6:
        return
    ratio = x / s
    if np.array_equal(q, np.clip(np.round(x * s), -127, 127)):
        fail("You multiplied `w` by `scale`. Going from small floats to big integers, which way should you go?")
    if not np.allclose(q, np.round(q)):
        fail("`q` still has decimals. Snap each value to a whole number first.")
    for fn, what in ((np.floor, "`floor()` always rounds down"), (np.ceil, "`ceil()` always rounds up"),
                     (np.trunc, "Casting straight to int8 chops off the decimals")):
        if np.array_equal(q, np.clip(fn(ratio), -127, 127)):
            fail(f"{what}, so the error can reach a whole step. Which rounding goes to the *nearest* integer?")
    fail(f"max err is {err:.4f} but the bound is {s / 2:.4f}. Fix the tests above first.")
