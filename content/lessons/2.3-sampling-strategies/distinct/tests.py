from harness import test, fail, warn, learner_fn, as_float, script_error, stdout, LearnerError

# (ids, distinct-2) with the pairs worked out, so a wrong formula shows up as a specific wrong number.
CASES = [
    ([1, 2, 1, 2, 1, 2], 2 / 5),   # pairs (1,2) (2,1) (1,2) (2,1) (1,2): 2 unique of 5
    ([5, 5, 5, 5], 1 / 3),         # (5,5) three times: 1 unique of 3
    ([1, 2, 3, 4], 1.0),           # 3 unique of 3
    ([3, 1, 4, 1, 5, 9, 2, 6], 1.0),
    ([7, 7, 8, 7, 7, 8], 3 / 5),   # (7,7) (7,8) (8,7) (7,7) (7,8): 3 unique of 5
]


def pairs_of(ids, gap=1):
    return list(zip(ids, ids[gap:]))


WRONG = {
    "distinct_1": (lambda ids: len(set(ids)) / len(ids),
                   "That's distinct-1, the fraction of unique *ids*. distinct-2 looks at neighbouring pairs: is each pair "
                   "`(ids[i], ids[i + 1])`?"),
    "per_token": (lambda ids: len(set(pairs_of(ids))) / len(ids),
                  "You divided by the number of ids, but a sequence of n ids has n − 1 neighbouring pairs. "
                  "What should the unique pairs be a fraction of?"),
    "count": (lambda ids: float(len(set(pairs_of(ids)))),
              "That's the number of unique pairs. distinct-2 is a fraction, so it stays between 0 and 1 however long the text is."),
    "inverted": (lambda ids: len(pairs_of(ids)) / len(set(pairs_of(ids))),
                 "That's all pairs divided by unique pairs, upside down: it grows as the text gets *more* repetitive."),
    "gap2": (lambda ids: len(set(pairs_of(ids, 2))) / len(pairs_of(ids, 2)),
             "Your pairs skip a token: `ids[2:]` pairs each id with the one two places later. Neighbours are one place apart."),
}


def check_steps(L):
    """Warn (don't fail) when an earlier step's function has drifted since it was solved."""
    import torch
    from harness import arr
    try:
        a = arr(L["top_k"](torch.tensor([1.0, 4.0, 2.0, 3.0]), 2))
        b = arr(L["top_p"](torch.tensor([3.0, 0.0, 1.0]), 0.5))
        ok = a is not None and b is not None and int((a > -1e30).sum()) == 2 and int((b > -1e30).sum()) == 1
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: top_k or top_p no longer passes its step's tests")


def score(L, ids):
    fn = learner_fn(L, "distinct_2")
    try:
        out = fn(list(ids))
    except LearnerError as e:
        if "unhashable" in str(e):
            fail("`set()` refused your pairs: lists can't go in a set, because they can change. "
                 "Make each pair a tuple. `zip` gives tuples for free.")
        raise
    if out is Ellipsis:
        fail("`distinct_2` still returns `...`, the placeholder.")
    v = as_float(out)
    if v is None:
        fail(f"`distinct_2` should return a number, but it returns a {type(out).__name__}.")
    return v


@test("fraction", "distinct_2 is the fraction of neighbouring pairs that are unique")
def _(L):
    check_steps(L)
    for ids, want in CASES:
        got = score(L, ids)
        if abs(got - want) < 1e-9:
            continue
        for key, (fn, story) in WRONG.items():
            if all(abs(score(L, c) - fn(c)) < 1e-9 for c, _ in CASES):
                fail(f"For `{ids}` you got {got:.2f}, but distinct-2 is {want:.2f}. {story}")
        fail(f"For `{ids}` you got {got:.2f}, but distinct-2 is {want:.2f}: "
             f"{len(set(pairs_of(ids)))} unique pairs out of {len(ids) - 1}.")


@test("script_runs", "The script compares four strategies")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if stdout().count("distinct-2") < 4:
        fail("The loop that runs and scores the four strategies is missing. Put it back at the bottom.")
