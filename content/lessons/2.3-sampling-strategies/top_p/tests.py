import numpy as np
import torch
from harness import test, fail, warn, learner_fn, arr, script_error, stdout

ROW = [0.3, 1.7, -0.4, 2.2, 0.9, -1.3, 1.1, 0.6]
PS = (0.3, 0.6, 0.8, 0.95)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def nucleus(p, rule="right", values=ROW):
    """The set of positions kept under the right rule, or under one of the usual wrong ones."""
    x = np.array(values, dtype=np.float32).astype(np.float64)
    probs = softmax(x)
    order = np.argsort(-probs, kind="stable")
    s = probs[order]
    cum = np.cumsum(s)
    cut_sorted = {
        "right": cum - s >= p,
        "cum_gt": cum > p,
        "flipped": cum - s < p,
        "unsorted": np.cumsum(probs) - s >= p,
        "own_prob": s < 1 - p,
    }[rule]
    return {int(i) for i, c in zip(order, cut_sorted) if not c}


def check_step1(L):
    """Warn (don't fail) when step 1's function has drifted since it was solved."""
    fn = L.get("top_k")
    if not callable(fn):
        return
    try:
        a = arr(fn(torch.tensor([1.0, 4.0, 2.0, 3.0]), 2))
        ok = a is not None and np.isneginf(a[[0, 2]]).all() and np.allclose(a[[1, 3]], [4.0, 3.0])
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: top_k no longer passes step 1's tests")


def kept_by(L, p, values=ROW):
    top_p = learner_fn(L, "top_p")
    out = top_p(torch.tensor(values), p)
    if out is Ellipsis:
        fail("`top_p` still returns `...`, the placeholder.")
    a = arr(out)
    if a is None:
        fail(f"`top_p` should return a tensor of logits, but it returns a {type(out).__name__}.")
    if a.shape != (len(values),):
        fail(f"`top_p` returned shape {list(a.shape)} for a row of {len(values)} logits. Keep one logit per token.")
    return {i for i in range(len(values)) if not np.isneginf(a[i])}


STORIES = {
    "cum_gt": "The token whose running total crosses p got cut. Top-p keeps that one: it's the token that makes "
              "the kept set reach p. Cut a token only when the tokens ranked *above* it already cover p.",
    "flipped": "You kept the unlikely tail and cut the most likely tokens. Which side of the cut is `True`?",
    "unsorted": "`cum` adds the probabilities up in vocabulary order. Top-p ranks the tokens first, "
                "so the running total has to run down the *sorted* probabilities.",
    "own_prob": "That cuts each token by its own probability. Top-p adds the probabilities up, from the most likely down, and stops at p.",
}


def same_rule(L, rule):
    return all(kept_by(L, p) == nucleus(p, rule) for p in PS)


@test("keeps_something", "top_p always keeps the most likely token")
def _(L):
    check_step1(L)
    top = softmax(np.array(ROW)).max()
    for p in PS:
        got = kept_by(L, p)
        if got and max(range(len(ROW)), key=lambda i: ROW[i]) in got:
            continue
        rule = next((r for r in ("cum_gt", "own_prob", "flipped") if same_rule(L, r)), None)
        what = "cut every token, and softmax over all `-inf` gives NaN" if not got else "cut the most likely token"
        story = STORIES[rule] if rule else ("Top-p keeps tokens from the top down until they cover p, so the most "
                                            f"likely token (probability {top:.2f}) always stays.")
        fail(f"With p = {p}, `top_p` {what}. {story}")


@test("nucleus", "Keeps the most likely tokens until they cover p")
def _(L):
    probs = softmax(np.array(ROW))
    for p in PS:
        got, want = kept_by(L, p), nucleus(p)
        if got == want:
            continue
        mass = probs[sorted(want)].sum()
        if got == set(range(len(ROW))):
            fail(f"With p = {p}, `top_p` kept every token. The {len(want)} most likely already cover {mass:.2f}. "
                 "Is `cum` a running total, so that it grows down the sorted list?")
        # Prefer a wrong rule that explains every p, then one that explains this p.
        rule = next((r for r in STORIES if same_rule(L, r)), None) or next((r for r in STORIES if got == nucleus(p, r)), None)
        if rule:
            fail(f"With p = {p} you kept {len(got)} tokens instead of {len(want)}. {STORIES[rule]}")
        fail(f"With p = {p} you kept positions {sorted(got)}, but the smallest set of top tokens covering {p} is "
             f"{sorted(want)} (together {mass:.2f}).")


@test("script_runs", "The script prints how many tokens each p keeps")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "top_p(row, 0.99)" not in stdout():
        fail("The loop that prints how many tokens each p keeps is missing. Put it back at the bottom.")
