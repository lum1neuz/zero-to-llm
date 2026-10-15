import numpy as np
import torch
from harness import test, fail, warn, learner_fn, arr, script_error, stdout

# Row 0 has 2 pads, row 1 none, row 2 has 4.
MASK = [[False, False, True, True, True, True],
        [True, True, True, True, True, True],
        [False, False, False, False, True, True]]
TEXTS = ["a cat", "the dog", "at"]


def check_step1(L):
    """Warn (don't fail) when step 1's function has drifted since it was solved."""
    fn = L.get("left_pad")
    if not callable(fn):
        return
    try:
        idx, mask = fn([torch.tensor([5, 6, 7]), torch.tensor([8])])
        ok = arr(idx).tolist() == [[5, 6, 7], [0, 0, 8]] and arr(mask).tolist() == [[True] * 3, [False, False, True]]
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: left_pad no longer passes step 1's tests")


def run(L, mask=MASK):
    batch_inputs = learner_fn(L, "batch_inputs")
    out = batch_inputs(torch.tensor(mask))
    if not isinstance(out, tuple) or len(out) != 2:
        fail("`batch_inputs` should return two things: `return pos, keep`.")
    pos, keep = out
    if pos is Ellipsis:
        fail("`pos` is still `...`, the placeholder.")
    if keep is Ellipsis:
        fail("`keep` is still `...`, the placeholder.")
    if arr(pos) is None or arr(keep) is None:
        fail("`batch_inputs` should return two tensors, `pos` and `keep`.")
    return arr(pos), arr(keep)


def row(values):
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


@test("positions", "Each row's first real token is position 0")
def _(L):
    check_step1(L)
    pos, _keep = run(L)
    m = np.array(MASK)
    B, T = m.shape
    if pos.shape != (B, T):
        fail(f"`pos` should be [{B}, {T}], one position per token, but it is {list(pos.shape)}.")
    if pos.dtype.kind not in "iu":
        fail(f"`pos` is {pos.dtype.name}. Positions index the position embedding, so they must be integers.")
    for b in range(B):
        real = pos[b][m[b]]
        pads = int((~m[b]).sum())
        want = list(range(len(real)))
        if real.tolist() == want and (pos[b] >= 0).all():
            continue
        if (pos[b] < 0).any():
            fail(f"Row {b} has {pads} pads and `pos` is {row(pos[b])}: the pads got negative positions, and the model's "
                 "position embedding can't look up −1. The pads' positions don't matter, as long as they're valid. How do you keep them at 0?")
        if real.tolist() == list(range(pads, pads + len(real))):
            fail(f"Row {b} has {pads} pads, so its first real token got position {pads} instead of 0, and the model sees "
                 "a different sequence than the same prompt alone. Count positions over real tokens only: "
                 "what running total of `mask` gives 1, 2, 3… on the real tokens?")
        if real.tolist() == list(range(1, len(real) + 1)):
            fail(f"Row {b}'s real tokens got positions {row(real)}: they start at 1. The first real token is position 0.")
        fail(f"Row {b}'s real tokens got positions {row(real)}, but they should be {row(want)}.")


@test("attention_mask", "keep: causal, and real queries never see pads")
def _(L):
    _pos, keep = run(L)
    m = np.array(MASK)
    B, T = m.shape
    if keep.shape == (T, T):
        fail(f"`keep` is [{T}, {T}], one mask shared by every row, but each row has its own padding. It should be [{B}, {T}, {T}].")
    if keep.shape != (B, T, T):
        fail(f"`keep` should be [{B}, {T}, {T}] (row, query, key), but it is {list(keep.shape)}.")
    if keep.dtype != np.bool_:
        fail(f"`keep` is {keep.dtype.name}. The model uses `~keep`, so it must be bool.")
    causal = np.tril(np.ones((T, T), bool))
    want = causal[None] & m[:, None, :]
    if np.array_equal(keep, want):
        return
    if (keep & ~causal[None]).any():
        b, t, s = map(int, np.argwhere(keep & ~causal[None])[0])
        fail(f"In row {b}, query {t} may attend to key {s}, which comes after it. Generation can't see the future, "
             "so keep still needs the causal mask, combined with the padding mask so both must hold.")
    if np.array_equal(keep, np.broadcast_to(causal, (B, T, T))):
        fail("`keep` is the plain causal mask, so real tokens attend to the pads in front of them. Pad embeddings are real "
             "vectors and they change the output. Which keys should be hidden, in every row?")
    if np.array_equal(keep, causal[None] & m[:, :, None]):
        fail("`keep` hides the pad *queries* (rows) but not the pad *keys* (columns), so real tokens still attend to pads. "
             "`keep[b, t, s]` is about key s. Which index of `mask` lines up with the key dimension?")
    if (want & ~keep).any():
        b, t, s = map(int, np.argwhere(want & ~keep)[0])
        fail(f"In row {b}, query {t} can't attend to key {s}, a real token before it. Only pads and future keys should be hidden.")
    b, t, s = map(int, np.argwhere(keep & ~want)[0])
    fail(f"In row {b}, query {t} may attend to key {s}, which is a pad. Real tokens must never see pads.")


@test("matches_single", "Batched logits match each prompt run alone")
def _(L):
    TinyLM, encode = L.get("TinyLM"), L.get("encode")
    if TinyLM is None or encode is None:
        fail("`TinyLM` or `encode` is gone from the top of the file. Put the model code back.")
    m = TinyLM(seed=2)
    prompts = [encode(t) for t in TEXTS]
    T = max(len(p) for p in prompts)
    idx = torch.zeros(len(prompts), T, dtype=torch.long)
    mask = torch.zeros(len(prompts), T, dtype=torch.bool)
    for b, p in enumerate(prompts):
        idx[b, T - len(p):] = p
        mask[b, T - len(p):] = True
    pos, keep = run(L, arr(mask).tolist())
    try:
        batched = arr(m(idx, torch.tensor(pos), torch.tensor(keep)))
    except Exception as e:  # noqa: BLE001 - invalid pos/keep make the model itself raise
        fail(f"The model couldn't run with your `pos` and `keep` ({type(e).__name__}: {e}). Fix the tests above first.")
    for b, p in enumerate(prompts):
        alone = arr(m(p[None]))[0]
        diff = float(np.abs(batched[b, T - len(p):] - alone).max())
        if diff > 1e-4:
            fail(f"Prompt {b}, \"{TEXTS[b]}\", gives logits up to {diff:.3f} away from the same prompt run alone. "
                 "Padding must not change anything for the real tokens.")


@test("script_runs", "Batched generation matches each prompt alone")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    out = stdout()
    if out.count("same as alone") < 3:
        fail("The loop that compares batched and single generation is missing. Put it back at the bottom.")
    if "same as alone: False" in out:
        fail("At least one row of the batch wrote different text from the same prompt alone. Check the printout.")
