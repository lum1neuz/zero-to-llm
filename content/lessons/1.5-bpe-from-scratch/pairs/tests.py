from harness import test, fail, learner_fn, LearnerError


def stats_of(L, ids):
    fn = learner_fn(L, "get_stats")
    try:
        out = fn(list(ids))
    except LearnerError as e:
        if "out of range" in str(e):
            fail("`get_stats` reads past the end of `ids`. The last id has no neighbour after it, so there's one pair fewer than ids.")
        if "is still `...`" in str(e):
            fail("`get_stats` still loops over `...`, the placeholder. Which pairs of neighbours should it loop over?")
        raise
    if not isinstance(out, dict):
        fail(f"`get_stats` should return a dict of pair → count, but it returns {type(out).__name__}.")
    return out


def merged(L, ids, pair, idx):
    fn = learner_fn(L, "merge")
    before = list(ids)
    arg = list(ids)
    try:
        out = fn(arg, pair, idx)
    except LearnerError as e:
        if "out of range" in str(e):
            fail(f"`merge({before}, {pair}, {idx})` reads past the end of the list. When `i` is the last position, "
                 "there's no `ids[i + 1]`, so check that a next id exists before you compare it.")
        raise
    if arg != before:
        fail("`merge` changed the list it was given. Build the result in `out` and leave `ids` alone.")
    if not isinstance(out, list):
        fail(f"`merge` should return a list of ids, but it returns {type(out).__name__}.")
    return out


@test("counts_pairs", "get_stats counts every pair of neighbours")
def _(L):
    ids = [1, 2, 3, 1, 2]
    got = stats_of(L, ids)
    want = {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    if got == want:
        return
    if any(v is Ellipsis for v in got.values()):
        fail("The counts are still `...`, the placeholder. Each time a pair shows up, its count should go up by one.")
    if set(got) == set(want) and all(v == 1 for v in got.values()):
        fail("Every pair has a count of 1, but (1, 2) appears twice in [1, 2, 3, 1, 2]. You set the count instead of adding to it.")
    if (1, 1) in got or (2, 2) in got:
        fail(f"Your pairs include {next(p for p in got if p[0] == p[1])}: each id paired with itself. Pair each id with the *next* one.")
    if set(got) < set(want):
        missing = sorted(set(want) - set(got))
        fail(f"For [1, 2, 3, 1, 2] you missed {missing}. Neighbours overlap: 2 belongs to both (1, 2) and (2, 3), so every position starts a pair.")
    fail(f"For [1, 2, 3, 1, 2] you count {got}. It should be {want}.")


@test("merges_pair", "merge writes idx for each occurrence of the pair, left to right")
def _(L):
    got = merged(L, [1, 2, 3, 1, 2], (1, 2), 99)
    if got == [99, 99, 99]:
        fail("Everything got merged: [1, 2, 3, 1, 2] became [99, 99, 99]. The `if` is still `...`, which Python treats as True. "
             "It should check that the ids at i and i + 1 are the pair.")
    if got == [99, 2, 3, 99, 2]:
        fail("[1, 2, 3, 1, 2] became [99, 2, 3, 99, 2]: the pair's second id is still there. After writing idx, skip past both ids.")
    if got != [99, 3, 99]:
        fail(f"`merge([1, 2, 3, 1, 2], (1, 2), 99)` gives {got}. It should be [99, 3, 99].")
    got = merged(L, [1, 3, 1, 2], (1, 2), 99)
    if got == [99, 99]:
        fail("[1, 3, 1, 2] became [99, 99]: you merged (1, 3) too. Check both ids of the pair, not only the first.")
    if got != [1, 3, 99]:
        fail(f"`merge([1, 3, 1, 2], (1, 2), 99)` gives {got}. It should be [1, 3, 99].")
    got = merged(L, [7, 7, 7, 7, 7], (7, 7), 9)
    if got != [9, 9, 7]:
        fail(f"`merge([7, 7, 7, 7, 7], (7, 7), 9)` gives {got}. Going left to right, the first two 7s merge, then the next two, "
             "and the fifth is left over: [9, 9, 7].")
    got = merged(L, [2, 1], (1, 2), 99)
    if got != [2, 1]:
        fail(f"`merge([2, 1], (1, 2), 99)` gives {got}, but (1, 2) never appears in [2, 1].")
