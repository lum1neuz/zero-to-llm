import numpy as np
import torch
from harness import test, fail, warn, learner_fn, arr, script_error, stdout

W = [0.12, -0.31, 0.05, 0.22, -0.18, 0.09, 2.40, -0.27, 0.15, -0.08, 0.30, -0.11, -0.25, 0.19, -0.04, 0.28]
# Same shape as the script's last run (4 groups of 4) but different scales, so code that reads the
# script's global `scale` instead of its argument can't pass by accident.
ROW_B = [0.50, -0.20, 0.10, 0.05, -1.00, 0.30, 0.20, -0.10, 0.06, 0.02, -0.04, 0.01, 0.90, -0.70, 0.35, 0.03]
W24 = [0.21, -0.64, 0.10, 0.33, -0.05, 0.47, -0.29, 0.12,
       1.80, -0.40, 0.25, 0.90, -0.15, 0.62, -1.10, 0.07,
       -0.02, 0.09, -0.13, 0.04, 0.11, -0.06, 0.08, -0.01]


def shape(a):
    return "[" + ", ".join(str(d) for d in a.shape) + "]"


def reference_quantize(values, g):
    x = np.array(values, dtype=np.float32).reshape(-1, g)
    s = (np.abs(x).max(axis=1, keepdims=True) / np.float32(7)).astype(np.float32)
    q = np.clip(np.round(x / s), -7, 7).astype(np.int8)
    return q, s


def check_step1(L):
    """Warn (don't fail) when step 1's function has drifted since it was solved."""
    fn = L.get("quantize_groups")
    if not callable(fn):
        return
    try:
        q, s = fn(torch.tensor(W24), 8)
        _, ref = reference_quantize(W24, 8)
        ok = arr(q) is not None and arr(q).dtype == np.int8 and arr(s) is not None and np.allclose(arr(s), ref, rtol=1e-4)
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: quantize_groups no longer passes step 1's tests")


def learner_round_trip(L, values, g):
    dequantize = learner_fn(L, "dequantize_groups")
    q, s = reference_quantize(values, g)
    out = dequantize(torch.tensor(q.tolist(), dtype=torch.int8), torch.tensor(s.tolist()))
    if out is None:
        fail("`dequantize_groups` returns nothing. Return one expression built from `q` and `scale`.")
    if out is Ellipsis:
        fail("`dequantize_groups` still returns `...`, the placeholder.")
    wa = arr(out)
    if wa is None:
        fail(f"`dequantize_groups` should return a tensor, but it returns {type(out).__name__}.")
    return q, s, wa


@test("returns_float", "dequantize_groups returns floats built from q and scale")
def _(L):
    check_step1(L)
    q, s, wa = learner_round_trip(L, W, 4)
    if wa.dtype.kind != "f":
        fail(f"`w_hat` comes back as {wa.dtype.name}, so it can only hold whole numbers. Turn `q` into floats first.")
    if wa.size == q.size and np.allclose(wa.reshape(-1), q.reshape(-1).astype(np.float32)):
        fail("`w_hat` is just `q` as floats. The integers are still in units of each group's `scale`. What brings them back to the weights' size?")


@test("flat_row", "w_hat comes back as one row, like w")
def _(L):
    q, s, wa = learner_round_trip(L, W, 4)
    if wa.shape == (16,):
        return
    if wa.shape == (4, 4):
        fail("`w_hat` still has shape [4, 4], one row per group. Flatten it back into one row of 16 so it lines up with `w`.")
    if wa.size == 4 * 16:
        fail(f"`w_hat` has shape {shape(wa)}: you flattened `q` before multiplying, so each of the 4 scales multiplied the whole row. "
             "Multiply while `q` is [4, 4] and `scale` is [4, 1], then flatten.")
    fail(f"`w_hat` has shape {shape(wa)}, but `w` has shape [16]. Return one value per weight, in one row.")


@test("within_bound", "Each weight is within its group's scale / 2")
def _(L):
    for values, g in ((ROW_B, 4), (W24, 8)):
        q, s, wa = learner_round_trip(L, values, g)
        x = np.array(values, dtype=np.float32)
        if wa.shape != x.shape:
            fail(f"On {len(values)} weights in groups of {g}, `w_hat` has shape {shape(wa)} instead of [{len(values)}]. Flatten after multiplying.")
        bound = np.repeat(s.reshape(-1), g) / 2
        err = np.abs(x - wa)
        if (err <= bound + 1e-6).all():
            continue
        if np.allclose(wa, (q / s).reshape(-1), rtol=1e-4):
            fail(f"You divided by `scale` again, so `w_hat` is off by up to {err.max():.1f}. Quantizing divided; undoing it does the opposite.")
        s_global = arr(L.get("scale")) if isinstance(L.get("scale"), torch.Tensor) else None
        if s_global is not None and s_global.shape == s.shape and np.allclose(wa, (q * s_global).reshape(-1), rtol=1e-4):
            fail("`dequantize_groups` ignores the scale it's given and uses `scale` from your script's loop. "
                 "On another row with different scales, `w_hat` comes out wrong. Use the function's own `scale` argument.")
        i = int(np.argmax(err - bound))
        fail(f"Weight {i} (group {i // g}) is off by {err[i]:.4f}, over its bound of {bound[i]:.4f}. "
             "Multiply each group's integers by that group's `scale`.")


@test("script_runs", "The script runs and prints both group sizes")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "w_hat" not in L:
        fail("The loop that calls `quantize_groups` and `dequantize_groups` is missing. Put it back at the bottom.")
    if not stdout().strip():
        fail("Nothing printed. Add the print line back so you can compare the two group sizes.")
