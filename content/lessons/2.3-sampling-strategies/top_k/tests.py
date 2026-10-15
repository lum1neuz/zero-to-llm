import numpy as np
import torch
from harness import test, fail, learner_fn, arr, script_error, stdout

ROW = [0.3, 1.7, -0.4, 2.2, 0.9, -1.3, 1.1, 0.6]
SHOWN = "`[0.3, 1.7, -0.4, 2.2, 0.9, -1.3, 1.1, 0.6]`"


def run(L, k, values=ROW):
    top_k = learner_fn(L, "top_k")
    out = top_k(torch.tensor(values), k)
    if out is Ellipsis:
        fail("`top_k` still returns `...`, the placeholder.")
    if out is None:
        fail("`top_k` returns nothing. Return the filtered logits.")
    a = arr(out)
    if a is None:
        fail(f"`top_k` should return a tensor of logits, but it returns a {type(out).__name__}.")
    if a.shape != (len(values),):
        if a.shape == (k,):
            fail(f"`top_k(row, {k})` returned only the {k} largest values. The sampler needs one logit per token, "
                 "so keep the row's shape and push the cut tokens to `-inf` instead.")
        fail(f"`top_k` returned shape {list(a.shape)} for a row of {len(values)} logits. Keep one logit per token.")
    return a.astype(np.float64)


def nth(k):
    return f"{k}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(k, 'th')}"


def kept(a, values=ROW):
    """Positions whose logit came back unchanged."""
    x = np.array(values, dtype=np.float32).astype(np.float64)
    return {i for i in range(len(x)) if np.isclose(a[i], x[i])}


@test("same_shape", "top_k returns one logit per token")
def _(L):
    run(L, 3)


@test("keeps_top_k", "The k largest logits survive, unchanged")
def _(L):
    x = np.array(ROW)
    order = list(np.argsort(-x))
    for k in (3, 1, 5):
        a = run(L, k)
        got, want = kept(a), set(order[:k])
        if got == want:
            continue
        if not got:
            fail(f"`top_k(row, {k})` cut every token: all {len(ROW)} logits changed. `kth` should be a logit *value* "
                 "from the row (the k-th largest), and only logits below it get cut. Print `kth` and check what it holds.")
        if got == set(order[:1]) and k > 1:
            fail(f"`top_k(row, {k})` kept only 1 token, the largest. `kth` is the biggest of the top {k}, "
                 f"but the cutoff is the *smallest* of them, the {nth(k)} largest.")
        if got == set(order[: k - 1]):
            fail(f"`top_k(row, {k})` kept {k - 1} tokens: the {nth(k)} largest was cut too. Only logits *strictly below* `kth` go.")
        if got == set(order[k - 1:]) or got == set(order[k:]):
            fail(f"On {SHOWN} with k = {k}, you kept the *smallest* logits and cut the largest. Which side of `kth` gets cut?")
        fail(f"On {SHOWN} with k = {k}, you kept positions {sorted(got)}, but the {k} largest logits sit at {sorted(want)}.")


@test("cut_to_zero_prob", "Cut tokens get probability 0")
def _(L):
    x = np.array(ROW)
    a = run(L, 3)
    cut = [i for i in range(len(ROW)) if i not in set(np.argsort(-x)[:3])]
    vals = a[cut]
    if np.all(np.isneginf(vals)):
        return
    if np.allclose(vals, 0.0):
        p = np.exp(a - a.max())
        p /= p.sum()
        fail(f"The cut tokens got the logit 0, but 0 is an ordinary logit: after softmax the 5 cut tokens still hold "
             f"{p[cut].sum():.0%} of the probability. Which logit makes softmax give exactly 0?")
    fail(f"The cut tokens got the logits {', '.join(f'{v:g}' for v in vals)}. They need a logit that softmax turns into exactly 0.")


@test("script_runs", "The script prints the filtered row and its probabilities")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "probs:" not in stdout():
        fail("The lines that print `top_k(row, 3)` and its probabilities are missing. Put them back at the bottom.")
