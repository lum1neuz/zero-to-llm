import numpy as np
import torch
from harness import test, fail, warn, learner_fn, arr, script_error, stdout


def lesson_parts():
    """The 8 sub-block scales and mins (as positive numbers) of the lesson's super-block."""
    wb = (0.05 * torch.randn(256, generator=torch.Generator().manual_seed(0))).reshape(-1, 32)
    lo = wb.amin(dim=1, keepdim=True)
    hi = wb.amax(dim=1, keepdim=True)
    return (hi - lo) / 15, -lo


SCALES, MINS = lesson_parts()
OTHER = torch.tensor([0.30, 1.20, 0.05, 0.90, 0.60])   # another length and range


def check_step1(L):
    fn = L.get("quantize_subblocks")
    if not callable(fn):
        return
    try:
        _q, s, _lo = fn(0.05 * torch.randn(256, generator=torch.Generator().manual_seed(0)))
        ok = np.allclose(arr(s), arr(SCALES), rtol=1e-5)
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: quantize_subblocks no longer passes step 1's tests")


def call(L, v):
    fn = learner_fn(L, "six_bit")
    out = fn(v)
    if not isinstance(out, tuple) or len(out) != 2:
        fail("`six_bit` should return two things: `return q.to(torch.uint8), d`.")
    q, d = out
    if d is Ellipsis:
        fail("`d` is still `...`, the placeholder. Start with the super-scale.")
    qa, da = arr(q), arr(d)
    if qa is None or da is None:
        fail("`six_bit` should return two tensors, the 6-bit values and the super-scale.")
    return qa.astype(np.float64), da.astype(np.float64)


@test("largest_63", "One super-scale; the largest value lands on 63")
def _(L):
    check_step1(L)
    for v in (SCALES, MINS, OTHER):
        qa, da = call(L, v)
        x = arr(v).astype(np.float64)
        if da.size != 1:
            fail(f"`d` has shape [{', '.join(map(str, da.shape))}], one per value, so every value lands on 63 and the values "
                 "tell you nothing. Use one super-scale for all of them: take the largest value, then divide.")
        d = float(da.reshape(-1)[0])
        top = x.max()
        near = lambda a: np.isclose(d, a, rtol=1e-4)
        if near(top / 63):
            continue
        if near(top / 64):
            fail("Dividing by 64 puts the largest value on 64. 6 bits hold 0…63, so it gets cut back to 63, a whole step short. "
                 "Which number should the largest value land on?")
        if near(top / 31):
            fail("Dividing by 31 maps the largest value to 31, the top of 5 bits. Q4_K stores its scales and mins in 6 bits.")
        if near(top / 127) or near(top / 255):
            fail("That's an 8-bit range. Q4_K stores its scales and mins in 6 bits.")
        if near(63 / top):
            fail("Your super-scale is 63 divided by the largest value. Flip it, so that `v / d` turns the largest value into 63.")
        fail(f"`d` is {d:.6g}, but the largest value is {top:.6g}, so it lands on {top / d:.1f} instead of 63.")


@test("within_half_step", "Every value comes back within d / 2")
def _(L):
    for v in (SCALES, MINS, OTHER):
        qa, da = call(L, v)
        x = arr(v).astype(np.float64).reshape(qa.shape) if arr(v).size == qa.size else None
        if x is None:
            fail(f"`six_bit` returns {qa.size} values for {arr(v).size} inputs. Return one 6-bit value per input.")
        d = float(da.reshape(-1)[0])
        if (np.abs(x - qa * d) <= d / 2 + 1e-9).all():
            continue
        if np.array_equal(qa, np.floor(x / d + 1e-9)):
            fail("Casting straight to uint8 chops off the decimals, so 41.8 becomes 41 instead of 42. Round to the nearest whole number first.")
        i = int(np.argmax(np.abs(x - qa * d).reshape(-1)))
        fail(f"Value {i} comes back {np.abs(x - qa * d).reshape(-1)[i]:.3g} away, more than half a step ({d / 2:.3g}).")


@test("script_runs", "The script runs and prints both formats")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "Q4_K" not in stdout():
        fail("Nothing printed for Q4_K. Put the comparison lines back at the bottom.")
