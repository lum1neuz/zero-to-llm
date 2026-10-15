import numpy as np
import torch
from harness import test, fail, warn, learner_fn, arr, as_float, script_error, stdout

W = [0.42, -1.30, 0.07, 0.88, -0.25, 2.10, -0.61, 0.33]
OUTLIER = [-3.0, 1.0, 0.5, -0.2]


def reference_quantize(values):
    x = np.array(values, dtype=np.float32)
    s = np.float32(np.abs(x).max() / np.float32(127))
    q = np.clip(np.round(x / s), -127, 127).astype(np.int8)
    return q, float(s)


def check_step1(L):
    """Warn (don't fail) when step 1's function has drifted since it was solved."""
    fn = L.get("quantize_absmax")
    if not callable(fn):
        return
    try:
        q, s = fn(torch.tensor(OUTLIER))
        ok = arr(q) is not None and arr(q).dtype == np.int8 and abs(as_float(s) - 3.0 / 127) < 1e-4
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: quantize_absmax no longer passes step 1's tests")


def learner_round_trip(L, values=W):
    dequantize = learner_fn(L, "dequantize")
    q, s = reference_quantize(values)
    out = dequantize(torch.tensor(q.tolist(), dtype=torch.int8), torch.tensor(s))
    if out is None:
        fail("`dequantize` returns nothing. Return one expression built from `q` and `scale`.")
    if out is Ellipsis:
        fail("`dequantize` still returns `...`, the placeholder.")
    wa = arr(out)
    if wa is None:
        fail(f"`dequantize` should return a tensor, but it returns {type(out).__name__}.")
    return q, s, wa


@test("returns_float", "dequantize returns floats built from q and scale")
def _(L):
    check_step1(L)
    q, s, wa = learner_round_trip(L)
    if wa.dtype.kind != "f":
        fail(f"`w_hat` comes back as {wa.dtype.name}, so it can only hold whole numbers. Turn `q` into floats first.")
    if wa.shape != q.shape:
        fail(f"`w_hat` has shape {list(wa.shape)} but `q` has shape {list(q.shape)}. Return one value per weight.")
    if np.allclose(wa, q.astype(np.float32)):
        fail("`w_hat` is just `q` as floats. The integers are still in units of `scale`. What brings them back to the weights' size?")


@test("within_bound", "w_hat is within scale / 2 of w")
def _(L):
    # Two rows with different scales, so code that reads the script's global `scale` or `w`
    # instead of its arguments can't pass by accident.
    for values in (W, OUTLIER):
        q, s, wa = learner_round_trip(L, values)
        x = np.array(values, dtype=np.float32)
        err = float(np.abs(x - wa).max())
        if err <= s / 2 + 1e-6:
            continue
        if np.allclose(wa, q / s, rtol=1e-4):
            fail(f"You divided by `scale` again, so `w_hat` is off by up to {err:.1f}. Quantizing divided; undoing it does the opposite.")
        _, s_global = reference_quantize(W)
        if values is OUTLIER and np.allclose(wa, q.astype(np.float32) * s_global, rtol=1e-4):
            fail("`dequantize` ignores the scale it's given and uses the one from your script. "
                 "On another row with a different scale, `w_hat` comes out wrong. Use the function's own `scale` argument.")
        fail(f"max err {err:.4f} is over the bound of {s / 2:.4f}. Multiply the integers by the same `scale` you divided by.")


@test("script_runs", "The script runs and prints")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "w_hat" not in L:
        fail("The lines that call `quantize_absmax` and `dequantize` are missing. Put them back at the bottom.")
    if not stdout().strip():
        fail("Nothing printed. Add the print lines back so you can see the result.")
