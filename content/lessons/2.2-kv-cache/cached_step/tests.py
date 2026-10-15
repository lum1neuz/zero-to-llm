import numpy as np
import torch
from harness import test, fail, learner_fn, arr, script_error, stdout, LearnerError

TEXT = "a dog sat"


def fresh(L):
    TinyLM, encode = L.get("TinyLM"), L.get("encode")
    if TinyLM is None or encode is None:
        fail("`TinyLM` or `encode` is gone from the top of the file. Put the model code back.")
    return TinyLM(seed=3), encode(TEXT)


def reference(m, idx):
    """k, v and three versions of the logits at every position, computed in NumPy."""
    a = lambda t: arr(t).astype(np.float64)
    ids = arr(idx)
    x = a(m.wte)[ids] + a(m.wpe)[: len(ids)]
    q, k, v = x @ a(m.Wq), x @ a(m.Wk), x @ a(m.Wv)
    C = m.C
    out = {"k": k, "v": v}
    for name, scores in (("right", q @ k.T / C**0.5), ("no_scale", q @ k.T), ("by_c", q @ k.T / C)):
        s = np.where(np.tril(np.ones((len(ids), len(ids)), bool)), scores, -np.inf)
        w = np.exp(s - s.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        out[name] = a(m.bigram)[ids] + (w @ v) @ a(m.Wo)
    raw = np.where(np.tril(np.ones((len(ids), len(ids)), bool)), q @ k.T / C**0.5, 0.0)
    out["raw"] = a(m.bigram)[ids] + (raw @ v) @ a(m.Wo)
    return out


def run_steps(L):
    """Feed TEXT through the learner's step() one token at a time. Returns logits per step and cache snapshots."""
    step = learner_fn(L, "step")
    m, idx = fresh(L)
    cache = {"k": torch.zeros(0, m.C), "v": torch.zeros(0, m.C)}
    logits, sizes = [], []
    for pos in range(len(idx)):
        try:
            out = step(m, idx[pos], pos, cache)
        except LearnerError as e:
            if "same number of dimensions" in str(e):
                fail(f"`torch.cat` refused to join the cache, shape [{pos}, {m.C}], with `k`, shape [{m.C}]: "
                     "a table and a single row of numbers. How do you turn `k` into a table with one row?")
            if cache["k"] is Ellipsis or cache["v"] is Ellipsis:
                name = "k" if cache["k"] is Ellipsis else "v"
                fail(f'`cache["{name}"]` is still `...`, the placeholder. Set it to the old rows with this token\'s row added.')
            raise
        for name in ("k", "v"):
            if cache[name] is Ellipsis:
                fail(f'`cache["{name}"]` is still `...`, the placeholder. Set it to the old rows with this token\'s row added.')
            if arr(cache[name]) is None:
                fail(f'`cache["{name}"]` should be a tensor, but it is a {type(cache[name]).__name__}.')
        if out is Ellipsis or arr(out) is None:
            fail(f"`step` returned {'`...`' if out is Ellipsis else type(out).__name__} instead of logits. Fill in `att` and `h` first.")
        logits.append(arr(out).astype(np.float64))
        sizes.append((arr(cache["k"]).copy(), arr(cache["v"]).copy()))
    return m, idx, np.stack(logits) if all(l.shape == logits[0].shape for l in logits) else logits, sizes


@test("cache_grows", "The cache gains one row of k and one row of v per token")
def _(L):
    m, idx, _logits, sizes = run_steps(L)
    for pos, (k, v) in enumerate(sizes):
        want = (pos + 1, m.C)
        for name, t in (("k", k), ("v", v)):
            if t.shape != want:
                if t.shape == (1, m.C) and pos > 0:
                    fail(f'After {pos + 1} tokens, `cache["{name}"]` holds 1 row: each step replaced the cache instead of adding to it. '
                         "Every later token needs the keys and values of all the earlier ones.")
                fail(f'After {pos + 1} tokens, `cache["{name}"]` has shape {list(t.shape)}, but it should be {list(want)}: one row per token so far.')


@test("cache_holds_kv", "Row t of the cache is token t's key (and value)")
def _(L):
    m, idx, _logits, sizes = run_steps(L)
    ref = reference(m, idx)
    k, v = sizes[-1]
    if not np.allclose(k, ref["k"], atol=1e-5):
        fail('The rows of `cache["k"]` aren\'t the keys of the tokens so far. Append exactly this step\'s `k`.')
    if not np.allclose(v, ref["v"], atol=1e-5):
        if np.allclose(v, ref["k"], atol=1e-5):
            fail('`cache["v"]` holds the keys, not the values: the second append adds `k` again. Which vector belongs in the value cache?')
        fail('The rows of `cache["v"]` aren\'t the values of the tokens so far. Append exactly this step\'s `v`.')


@test("logits_match", "Cached logits match the full forward at every position")
def _(L):
    m, idx, logits, _sizes = run_steps(L)
    ref = reference(m, idx)
    if isinstance(logits, list) or logits.shape != ref["right"].shape:
        shapes = sorted({tuple(l.shape) for l in (logits if isinstance(logits, list) else [logits[0]])})
        fail(f"Each step should return {ref['right'].shape[1]} logits, one per vocabulary entry, but `step` returned shape {list(shapes[0])}. "
             "Check that `att` holds one weight per cached row and `h` has one number per channel.")
    diff = np.abs(logits - ref["right"]).max()
    if diff < 1e-4:
        return
    close = lambda name: np.abs(logits - ref[name]).max() < 1e-4
    if close("no_scale"):
        fail(f"Your logits are off by up to {diff:.3f}: the scores skip the division by √C. The full forward divides `q · k` by "
             f"√C = {m.C**0.5:g} before softmax, so the cached step has to as well.")
    if close("by_c"):
        fail(f"Your logits are off by up to {diff:.3f}: the scores are divided by C = {m.C}. Attention scales by √C = {m.C**0.5:g}.")
    if close("raw"):
        fail(f"Your logits are off by up to {diff:.3f}: `att` is the raw scaled scores. They have to go through softmax over the "
             "cached rows to become weights that sum to 1.")
    first = int(np.argmax(np.abs(logits - ref["right"]).max(axis=1)))
    fail(f"At position {first} your logits differ from the full forward by {np.abs(logits[first] - ref['right'][first]).max():.4f}. "
         "The cached step must give the same answer as running every position at once.")


@test("script_runs", "The script prints a diff for every prompt position")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if stdout().count("max diff") < 4:
        fail("The loop that compares `step` with the full forward is missing. Put it back at the bottom.")
