import numpy as np
import torch
from harness import test, fail, learner_fn, arr

# The lesson's row: 16 weights in 4 groups of 4. Group 0's largest magnitude is negative (−0.31).
W = [0.12, -0.31, 0.05, 0.22, -0.18, 0.09, 2.40, -0.27, 0.15, -0.08, 0.30, -0.11, -0.25, 0.19, -0.04, 0.28]
# 24 weights in 3 groups of 8. Not square, so swapped reshapes and scales that line up with
# columns can't pass by accident.
W24 = [0.21, -0.64, 0.10, 0.33, -0.05, 0.47, -0.29, 0.12,
       1.80, -0.40, 0.25, 0.90, -0.15, 0.62, -1.10, 0.07,
       -0.02, 0.09, -0.13, 0.04, 0.11, -0.06, 0.08, -0.01]


def num(v):
    """Format like the lesson text: two decimals for floats, a real minus sign."""
    text = f"{v:.2f}" if isinstance(v, float) else str(v)
    return text.replace("-", "−")


def shape(a):
    return "[" + ", ".join(str(d) for d in a.shape) + "]"


def run_on(L, values, g):
    """Call the learner's quantize_groups and unpack it, explaining any shape mistakes."""
    quantize = learner_fn(L, "quantize_groups")
    out = quantize(torch.tensor(values), g)
    if out is None:
        fail("`quantize_groups` returns nothing. It has to end with `return q, scale`.")
    if not isinstance(out, tuple) or len(out) != 2:
        got = "one value" if not isinstance(out, tuple) else f"{len(out)} values"
        fail(f"`quantize_groups` returns {got}. It has to return two: `return q, scale`.")
    q, scale = out
    if q is Ellipsis and scale is Ellipsis:
        fail("`wg`, `scale` and `q` are still `...`, the placeholders. Start with `wg`.")
    if scale is Ellipsis:
        fail("`scale` is still `...`, the placeholder. Each group needs the number that maps its largest weight onto 7.")
    if q is Ellipsis:
        fail("`q` is still `...`, the placeholder. Build it from `wg` and `scale`.")
    qa = arr(q)
    if qa is None:
        fail(f"`q` should be a tensor, but it's {type(q).__name__}.")
    sa = arr(scale)
    if sa is None:
        if isinstance(scale, (int, float)):
            fail("`scale` is a plain Python number. Keep it a tensor with one value per group, so `wg / scale` can line them up.")
        fail(f"`scale` should be a tensor, but it's {type(scale).__name__}.")
    return qa, sa


def grouped(L, values, g):
    """run_on for the tests after one_scale_per_group, which need the shapes right first."""
    qa, sa = run_on(L, values, g)
    n = len(values) // g
    if qa.shape != (n, g) or sa.shape != (n, 1):
        fail("`q` and `scale` aren't one row and one scale per group yet. Fix \"q is one row per group\" above first.")
    return qa, sa


def groups(values, g):
    return np.array(values, dtype=np.float32).reshape(-1, g)


def scale_story(values, g, s):
    """Name the specific mistake behind wrong scales, from their values alone."""
    x = groups(values, g)
    s = s.reshape(-1).astype(np.float64)
    amax = np.abs(x).max(axis=1).astype(np.float64)
    near = lambda a, b: np.allclose(a, b, rtol=1e-4, atol=1e-7)
    if near(s, amax / 7):
        return None
    k = int(np.argmax(np.abs(s - amax / 7) / (amax / 7)))  # the group that's furthest off
    biggest = float(x[k][np.abs(x[k]).argmax()])
    lands = round(biggest / s[k]) if s[k] else float("inf")
    if near(s, amax / 127):
        return ("Your scales divide by 127, int8's limit from 3.1, so the biggest weight in each group lands on ±127. "
                "int4 only holds −8…7. Which number should the largest |w| map to?")
    if near(s, amax / 8):
        return "Dividing by 8 puts each group's biggest weight on ±8, and +8 doesn't fit in int4. What's the largest value int4 can hold?"
    if near(s, amax / 15) or near(s, amax / 16):
        return (f"The biggest weight lands on {num(lands)}. 15 is the top of *unsigned* 4-bit (0…15). "
                "Signed int4 runs −8…7. Which limit keeps the grid symmetric around zero?")
    if near(s, 7 / amax):
        return "Your scale is 7 divided by the max. Flip it, so that `wg / scale` turns each group's biggest weight into 7."
    if near(s, x.max(axis=1) / 7):
        return (f"In group {k}, {group_txt(x[k])}, your scale comes from {num(float(x[k].max()))}, not {num(biggest)}, "
                f"so {num(biggest)} lands on {num(lands)}. Which weight has the largest *magnitude*?")
    if near(s, np.abs(x).mean(axis=1) / 7) or near(s, np.abs(x.mean(axis=1)) / 7):
        return f"The mean doesn't bound anything, so the biggest weight in group {k} lands on {num(lands)}. Which statistic does?"
    if near(s, amax):
        return "Your scale is each group's max itself, so every weight lands between −1 and 1. Divide by the int4 limit too."
    return f"With your scales, the biggest weight in group {k} ({num(biggest)}) lands on {num(lands)} instead of ±7. Check how you compute `scale`."


def group_txt(group):
    return "`[" + ", ".join(num(float(v)) for v in group) + "]`"


@test("returns_int8", "Returns q and scale, with q stored as int8")
def _(L):
    qa, _s = run_on(L, W, 4)
    if qa.dtype != np.int8:
        name = {"float32": "float32, which is 4 bytes per weight", "int64": "int64, which is 8 bytes per weight"}.get(qa.dtype.name, qa.dtype.name)
        fail(f"`q` is still {name}. PyTorch has no int4 dtype, so int4 values live in the smallest one. Which dtype stores exactly one byte?")


@test("one_scale_per_group", "q is one row per group, with one scale per group")
def _(L):
    qa, sa = run_on(L, W, 4)
    if qa.shape == (16,):
        if sa.size == 1:
            fail("`q` has shape [16] and there's one scale for the whole row, like 3.1. "
                 "Lay the row out as groups first, one row of `g` weights per group, and give each row its own scale.")
        fail("`q` has shape [16], so the row was never split into groups. Reshape `w` into rows of `g` weights first.")
    if qa.shape != (4, 4):
        fail(f"`q` has shape {shape(qa)}, but 16 weights in groups of 4 make [4, 4], one row per group.")
    if sa.size == 1:
        fail("`scale` is a single number, so all 4 groups share it. That's one scale per row again. "
             "Take the largest magnitude of each group separately.")
    if sa.shape == (4,):
        fail("`scale` has shape [4], not [4, 1]. Without `keepdim=True`, `wg / scale` lines the 4 scales up with the 4 *columns*, "
             "so every group gets divided by the other groups' scales. Keep the reduced dimension as a column.")
    if sa.shape == (1, 4):
        fail("`scale` has shape [1, 4]: `dim=0` reduces down each column, so you get one scale per column, shared across all the groups. "
             "Reduce across each row instead, the dimension that runs along a group.")
    if sa.shape != (4, 1):
        fail(f"`scale` has shape {shape(sa)}, but 4 groups need a [4, 1] column, one scale per row of `wg`.")
    qa, sa = run_on(L, W24, 8)
    if qa.shape == (8, 3):
        fail("On 24 weights with g=8, `q` has shape [8, 3], but 3 groups of 8 make [3, 8]. "
             "`reshape(g, -1)` makes g rows. Each group should be one row of g neighbours, so g goes last.")
    if qa.shape != (3, 8) or sa.shape != (3, 1):
        fail(f"On 24 weights with g=8, `q` has shape {shape(qa)} and `scale` {shape(sa)}. "
             "3 groups of 8 should give [3, 8] and [3, 1]. Use the `g` you're given.")


@test("largest_is_7", "Each group's largest |w| lands on ±7")
def _(L):
    for values, g in ((W, 4), (W24, 8)):
        _q, sa = grouped(L, values, g)
        story = scale_story(values, g, sa)
        if story:
            fail(story)


@test("round_trip", "Each weight is within its group's scale / 2")
def _(L):
    for values, g in ((W, 4), (W24, 8)):
        qa, sa = grouped(L, values, g)
        x = groups(values, g)
        q = qa.astype(np.float64)
        s = sa.astype(np.float64)
        err = np.abs(x - q * s)
        if (err <= s / 2 + 1e-6).all():
            continue
        ratio = x / s
        if np.array_equal(q, np.clip(np.round(x * s), -7, 7)):
            fail("You multiplied the weights by `scale`. Going from small floats to integers up to 7, which way should you go?")
        if not np.allclose(q, np.round(q)):
            fail("`q` still has decimals. Snap each value to a whole number first.")
        for fn, what in ((np.floor, "`floor()` always rounds down"), (np.ceil, "`ceil()` always rounds up"),
                         (np.trunc, "Casting straight to int8 chops off the decimals")):
            if np.array_equal(q, np.clip(fn(ratio), -7, 7)):
                fail(f"{what}, so the error can reach a whole step. Which rounding goes to the *nearest* integer?")
        k = int(np.argmax((err - s / 2).max(axis=1)))
        fail(f"In group {k}, max err is {err[k].max():.4f} but the bound is {s[k, 0] / 2:.4f}. Fix the tests above first.")
