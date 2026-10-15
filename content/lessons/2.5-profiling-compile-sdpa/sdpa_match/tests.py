import numpy as np
import torch
import torch.nn.functional as F
from harness import test, fail, warn, learner_fn, arr, script_error, stdout, LearnerError

SHAPE = (2, 3, 6, 4)  # B, H, T, D: T and D differ, so swapped dims and the wrong scale both show


def inputs():
    g = torch.Generator().manual_seed(11)
    return tuple(torch.randn(*SHAPE, generator=g) for _ in range(3))


def softmax(x, axis):
    e = np.exp(x - np.nanmax(np.where(np.isneginf(x), np.nan, x), axis=axis, keepdims=True))
    e = np.where(np.isneginf(x), 0.0, e)
    return e / e.sum(axis=axis, keepdims=True)


def variant(q, k, v, scale=None, mask="causal", fill=-np.inf, axis=-1, mix=True):
    """Attention in NumPy, with the usual mistakes switchable, to name what the learner did."""
    q, k, v = (arr(t).astype(np.float64) for t in (q, k, v))
    T, D = q.shape[-2], q.shape[-1]
    s = q @ np.swapaxes(k, -1, -2) / (D**0.5 if scale is None else scale)
    keep = {"causal": np.tril(np.ones((T, T), bool)), "none": np.ones((T, T), bool), "past": np.triu(np.ones((T, T), bool))}[mask]
    s = np.where(keep, s, fill)
    w = softmax(s, axis)
    return w @ v if mix else w


def check_step1(L):
    """Warn (don't fail) when step 1's function has drifted since it was solved."""
    fn = L.get("decode_step_ms")
    if not callable(fn):
        return
    try:
        ms, bound = fn(8e9, 1, 2, peak_tflops=71, bandwidth_gbs=936)
        ok = abs(ms - 8e9 * 2 / 936e9 * 1e3) < 1e-6 and bound == "memory"
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: decode_step_ms no longer passes step 1's tests")


def run(L):
    attention = learner_fn(L, "attention")
    q, k, v = inputs()
    try:
        out = attention(q, k, v)
    except LearnerError as e:
        if any(s in str(e) for s in ("cannot be multiplied", "must match the size", "Expected size for first two dimensions")):
            fail("The scores matmul failed. On a 4-D tensor `k.T` reverses *all four* dims, [B, H, T, D] → [D, T, H, B]. "
                 "You only want to swap the last two. Which method swaps two chosen dims?")
        raise
    if out is Ellipsis:
        fail("`attention` still returns `...`, the placeholder.")
    a = arr(out)
    if a is None:
        fail(f"`attention` should return a tensor, but it returns a {type(out).__name__}.")
    return (q, k, v), a.astype(np.float64)


@test("output_shape", "attention returns one D-vector per query: [B, H, T, D]")
def _(L):
    check_step1(L)
    (q, k, v), a = run(L)
    B, H, T, D = SHAPE
    if a.shape == (B, H, T, T):
        fail(f"`attention` returned [{B}, {H}, {T}, {T}]: the attention weights. The output mixes the values with them: "
             "one D-wide vector per query.")
    if a.shape != SHAPE:
        fail(f"`attention` returned shape {list(a.shape)}, but it should be {list(SHAPE)}, like q.")


@test("matches_sdpa", "Same output as F.scaled_dot_product_attention(is_causal=True)")
def _(L):
    (q, k, v), a = run(L)
    if a.shape != SHAPE:
        fail(f"`attention` returned shape {list(a.shape)} instead of {list(SHAPE)}. Fix the shape first.")
    ref = arr(F.scaled_dot_product_attention(q, k, v, is_causal=True)).astype(np.float64)
    diff = np.abs(a - ref).max()
    if diff < 1e-5:
        return
    D = SHAPE[-1]
    close = lambda **kw: np.abs(a - variant(q, k, v, **kw)).max() < 1e-5
    if close(scale=1.0):
        fail(f"Your output is up to {diff:.3f} away from SDPA: the scores aren't scaled. SDPA divides by √D = {D**0.5:g}, "
             "which keeps softmax from saturating as D grows.")
    if close(scale=float(D)):
        fail(f"Your output is up to {diff:.3f} away from SDPA: the scores are divided by D = {D}. The scale is √D = {D**0.5:g}.")
    if close(mask="none"):
        fail(f"Your output is up to {diff:.3f} away from SDPA with `is_causal=True`: every query sees every key, "
             "including later ones. Hide the future.")
    if close(mask="past"):
        fail("Your mask hides the past and keeps the future: query t sees keys t, t+1, …. Causal attention keeps keys 0…t. "
             "Which triangle is that?")
    if close(fill=0.0):
        fail("The future scores were set to 0, but 0 is an ordinary score: after softmax those keys still get weight. "
             "Which value makes softmax give exactly 0?")
    if close(axis=-2):
        fail("Your softmax runs over the queries (dim −2), so each *key's* weights sum to 1. Each query's weights over "
             "the keys should sum to 1. Which dim holds the keys?")
    fail(f"Your output is up to {diff:.4f} away from SDPA's. Scale by √D, mask the future with -inf, softmax over the keys, then mix v.")


@test("script_runs", "The script compares with SDPA")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "max diff vs SDPA" not in stdout():
        fail("The lines that compare `attention` with SDPA are missing. Put them back at the bottom.")
