import torch
from harness import test, fail, warn, learner_fn, arr, script_error, stdout

TEXT = "a cat "
N_NEW = 20


def fresh(L, seed=5):
    TinyLM, encode = L.get("TinyLM"), L.get("encode")
    if TinyLM is None or encode is None:
        fail("`TinyLM` or `encode` is gone from the top of the file. Put the model code back.")
    return TinyLM(seed=seed), encode(TEXT)


def text_of(L, ids):
    decode = L.get("decode")
    try:
        return repr(decode(torch.tensor(ids)))
    except Exception:  # noqa: BLE001
        return " ".join(map(str, ids))


def check_step1(L):
    """Warn (don't fail) when step 1's function has drifted since it was solved."""
    step = L.get("step")
    if not callable(step):
        return
    try:
        m, idx = fresh(L, seed=3)
        cache = {"k": torch.zeros(0, m.C), "v": torch.zeros(0, m.C)}
        full = m(idx)
        ok = all(float((step(m, idx[p], p, cache) - full[p]).abs().max()) < 1e-4 for p in range(len(idx)))
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: step no longer passes step 1's tests")


def run(L):
    """Run the learner's generate_cached on a fresh model, recording every (token, position) fed to embed()."""
    gen = learner_fn(L, "generate_cached")
    m, idx = fresh(L)
    fed = []
    real_embed = m.embed

    def spy(token, pos):
        t, p = torch.as_tensor(token), torch.as_tensor(pos)
        if t.numel() == 1 and p.numel() == 1:
            fed.append((int(t), int(p)))
        return real_embed(token, pos)

    m.embed = spy
    out = gen(m, idx, N_NEW)
    if out is None:
        fail("`generate_cached` returns nothing. Keep the `return idx` at the end.")
    a = arr(out)
    if a is None:
        fail(f"`generate_cached` should return a tensor of ids, but it returns a {type(out).__name__}.")
    return m, idx, a.reshape(-1).tolist(), fed


@test("returns_tokens", "generate_cached returns the prompt plus one id per step")
def _(L):
    check_step1(L)
    m, idx, got, fed = run(L)
    want = len(idx) + N_NEW
    if len(got) != want:
        fail(f"With a {len(idx)}-token prompt and {N_NEW} steps, `generate_cached` should return {want} ids, but it returned {len(got)}.")
    if got[: len(idx)] != idx.tolist():
        fail("The output doesn't start with the prompt. Append new ids after it; don't change it.")


@test("matches_no_cache", "Same text as the no-cache loop")
def _(L):
    m, idx, got, fed = run(L)
    slow, _ = fresh(L)
    ref = idx
    for _ in range(N_NEW):
        ref = torch.cat([ref, slow(ref)[-1].argmax().view(1)])
    ref = ref.tolist()
    if got == ref:
        return
    T = len(idx)
    pre, new = fed[:T], fed[T:]
    if [p for _, p in pre] != list(range(T)):
        fail(f"During the prefill the prompt tokens ran at positions {[p for _, p in pre]}, but they sit at {list(range(T))}. "
             "Each prompt token has its own position, `pos`.")
    if [t for t, _ in pre] != idx.tolist():
        fail("During the prefill the model didn't see the prompt tokens in order. Run `idx[pos]` at position `pos`.")
    new_pos = [p for _, p in new]
    if new_pos and all(p == 0 for p in new_pos):
        fail("Every new token ran at position 0, so the model treats each one as if it were the first token. "
             "Each new token sits at the end of `idx`. Which index is that?")
    if new_pos[:3] == [T + 1, T + 2, T + 3]:
        fail(f"The first new token sits at position {T} after a {T}-token prompt, but your loop ran it at {T + 1}: "
             "after `torch.cat`, `len(idx)` counts the new token too. Which index does the last token have?")
    if len(new) > 1 and new[0][0] == idx.tolist()[-1]:
        fail("Each step ran the token *before* the one you just picked, so the cache got that token twice. "
             "Run `next_id`, the id you just picked.")
    first = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))
    fail(f"From {text_of(L, idx.tolist())} the no-cache loop writes {text_of(L, ref)}, but yours writes {text_of(L, got)}. "
         f"They split at id {first}. Each new token needs its own id, its own position and the same cache.")


@test("one_position_per_token", "Each new token costs one position, not the whole sequence")
def _(L):
    m, idx, got, fed = run(L)
    want = len(idx) + N_NEW
    if m.positions == want:
        return
    if m.positions > want:
        fail(f"Writing {N_NEW} tokens from a {len(idx)}-token prompt processed {m.positions} positions; with a cache it takes "
             f"{want}, one per token. Something still runs the whole sequence. Use `step` on the new token only.")
    fail(f"The model processed {m.positions} positions, but the prompt and the new tokens add up to {want}. "
         "Every prompt token goes through `step` once, and so does every new token.")


@test("script_runs", "The script compares both loops")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "same text:" not in stdout():
        fail("The lines that run both loops and compare them are missing. Put them back at the bottom.")
