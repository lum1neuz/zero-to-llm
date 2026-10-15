import numpy as np
import torch
from harness import test, fail, learner_fn, arr

LETTERS = "abcdefghijklmnopqrstuvwxyz"
STOI = {ch: i + 1 for i, ch in enumerate(LETTERS)}
STOI["."] = 0
SMALL = ["ab", "ba", "aab"]


def reference(words):
    N = np.zeros((27, 27), dtype=np.int64)
    for w in words:
        chs = "." + w + "."
        for a, b in zip(chs, chs[1:]):
            N[STOI[a], STOI[b]] += 1
    return N


def counts(L, words):
    out = learner_fn(L, "count_bigrams")(list(words))
    a = arr(out)
    if a is None:
        fail(f"`count_bigrams` should return the tensor N, but it returns {type(out).__name__}.")
    if a.shape != (27, 27):
        fail(f"N has shape {list(a.shape)}. It needs one row and one column per symbol: [27, 27].")
    return a


def check_overwrite(a, ref):
    if a.max() == 1 and ref.max() > 1:
        fail(f"Every cell is 0 or 1, but in {SMALL} the pair (\".\", \"a\") happens twice. "
             "`=` stores a 1 each time. What adds one to what's already there?")


@test("returns_counts", "Returns a [27, 27] table with something counted")
def _(L):
    a = counts(L, SMALL)
    if a.dtype.kind not in "iu":
        fail(f"N is {a.dtype.name}. Counts are whole numbers, so keep N as int64.")
    if not a.any():
        fail("N is all zeros. The loops run, but nothing adds to N. Which cell belongs to the pair (ch1, ch2), and what happens to it?")


@test("start_and_end", "\".\" starts and ends every name")
def _(L):
    a = counts(L, SMALL)
    ref = reference(SMALL)
    if not a[0].any():
        fail("Row \".\" is all zeros, so nothing ever follows the start marker. "
             "How would the model learn which letters names start with?")
    if not a[:, 0].any():
        fail("Column \".\" is all zeros, so no name ever ends. "
             "How would the model learn when to stop?")
    check_overwrite(a, ref)
    if a[0].sum() != ref[0].sum() or a[:, 0].sum() != ref[:, 0].sum():
        fail(f"For {SMALL}, \".\" is followed {int(a[0].sum())} times and follows {int(a[:, 0].sum())} times. "
             f"Each name should add one of each: {len(SMALL)} and {len(SMALL)}.")


@test("right_cells", "N[i, j] counts how often j follows i")
def _(L):
    a = counts(L, SMALL)
    ref = reference(SMALL)
    if np.array_equal(a, ref):
        return
    if np.array_equal(a, ref.T):
        fail("N is transposed. The row should be the first character and the column the one that comes next, "
             "so that row i of N is everything that can follow i.")
    check_overwrite(a, ref)
    i, j = np.argwhere(a != ref)[0]
    names = ".abcdefghijklmnopqrstuvwxyz"
    fail(f"For {SMALL}, N[{i}, {j}] (\"{names[i]}\" then \"{names[j]}\") is {int(a[i, j])} but should be {int(ref[i, j])}.")


@test("all_names", "Counts every pair in the lesson's names")
def _(L):
    words = L.get("words")
    if not isinstance(words, list) or not words:
        fail("`words` is missing from the top of your script.")
    a = counts(L, words)
    total = sum(len(w) + 1 for w in words)
    if int(a.sum()) != total:
        fail(f"N holds {int(a.sum())} pairs, but {len(words)} names make {total}: one per letter plus the final \".\".")
