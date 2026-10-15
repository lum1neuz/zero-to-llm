import math

import numpy as np
import torch
from harness import test, fail, warn, learner_fn, as_float

LETTERS = "abcdefghijklmnopqrstuvwxyz"
STOI = {ch: i + 1 for i, ch in enumerate(LETTERS)}
STOI["."] = 0
WORDS = ["ab", "ba", "aab"]


def table():
    """A random but fixed probability table, so every pair has a different probability."""
    g = torch.Generator().manual_seed(1234)
    P = torch.rand((27, 27), generator=g) + 0.1
    return P / P.sum(1, keepdim=True)


def pairs(words):
    for w in words:
        chs = "." + w + "."
        yield from zip(chs, chs[1:])


def check_step2(L):
    fn = L.get("sample_name")
    P = torch.zeros((27, 27))
    P[0, 1] = P[1, 0] = 1.0
    try:
        ok = callable(fn) and fn(P, torch.Generator().manual_seed(1)) == "a"
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: sample_name no longer passes step 2's tests")


def loss(L, P, words):
    out = learner_fn(L, "nll")(P, list(words))
    if out is Ellipsis:
        fail("`nll` still returns `...`, the placeholder. Turn the summed log-likelihood into an average loss.")
    v = as_float(out)
    if v is None:
        fail(f"`nll` should return a single number, but it returns {type(out).__name__}.")
    return v


@test("returns_number", "nll returns one number")
def _(L):
    check_step2(L)
    loss(L, table(), WORDS)


@test("matches_reference", "nll is the average negative log-likelihood")
def _(L):
    P = table()
    Pa = P.numpy().astype(np.float64)
    logs = [math.log(Pa[STOI[a], STOI[b]]) for a, b in pairs(WORDS)]
    n = len(logs)
    want = -sum(logs) / n
    v = loss(L, P, WORDS)
    if abs(v - want) < 1e-4:
        return
    story = f"On {WORDS} ({n} pairs) your loss is {v:.4f}, but it should be {want:.4f}. "
    if abs(v + want) < 1e-4:
        fail(story + "The sign is flipped. Each log-probability is negative, so their average is too. A loss should get smaller as the model gets better.")
    if abs(v - want * n) < 1e-3:
        fail(story + f"That's the total over all {n} pairs. Divide by how many pairs you scored, so the loss doesn't grow with the data.")
    if abs(v - want * n / 27) < 1e-3:
        fail(story + "You divided by 27, the vocabulary size. The average is over the pairs you scored, and `n` counts them.")
    probs = [Pa[STOI[a], STOI[b]] for a, b in pairs(WORDS)]
    if abs(v + sum(probs) / n) < 1e-4 or abs(v - sum(probs) / n) < 1e-4:
        fail(story + "You averaged the probabilities themselves. The loss averages their *logs*, so one confident miss costs a lot.")
    swapped = -sum(math.log(Pa[STOI[b], STOI[a]]) for a, b in pairs(WORDS)) / n
    if abs(v - swapped) < 1e-4:
        fail(story + "You read P[ch2, ch1]. The row is the character you're coming from, ch1, and the column is the one that follows, ch2.")
    fail(story)


@test("beats_uniform", "The bigram model beats the uniform baseline on the lesson's names")
def _(L):
    P, words = L.get("P"), L.get("words")
    if P is None or words is None:
        fail("`P` or `words` is missing from your script. Keep the lines that build them.")
    v = loss(L, P, words)
    if not v < math.log(27):
        fail(f"Your loss on the lesson's names is {v:.4f}, no better than the uniform model's {math.log(27):.4f}. "
             "Fix the test above first.")
