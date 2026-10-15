import numpy as np
import torch
from harness import test, fail, warn, learner_fn, arr, script_error, stdout, LearnerError

LOGITS = [2.0, 1.0, 0.0, -1.0]
DRAWS = 800


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def check_step1(L):
    """Warn (don't fail) when step 1's loop has drifted since it was solved."""
    gen = L.get("generate")
    if not callable(gen):
        return
    try:
        out = gen(lambda idx: torch.tensor(np.eye(6, dtype=np.float32)[(arr(idx) + 1) % 6]), torch.tensor([0, 1]), 3)
        ok = arr(out) is not None and arr(out).tolist() == [0, 1, 2, 3, 4]
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: generate no longer passes step 1's tests")


def one_draw(L, logits, temperature, g):
    sample = learner_fn(L, "sample")
    try:
        out = sample(torch.tensor(logits), temperature, g)
    except LearnerError as e:
        if "element < 0" in str(e):
            fail("`torch.multinomial` refused `probs` because some entries are negative. `log_softmax` gives "
                 "log-probabilities, which are all ≤ 0. multinomial wants probabilities: which function gives those?")
        raise
    if out is Ellipsis:
        fail("`sample` still returns `...`, the placeholder.")
    if out is None:
        fail("`sample` returns nothing. It should return the id it drew.")
    a = arr(out)
    if a is None:
        fail(f"`sample` should return a tensor holding one id, but it returns a {type(out).__name__}.")
    return a


def draws(L, temperature, seed=0, n=DRAWS):
    g = torch.Generator().manual_seed(seed)
    return np.array([int(one_draw(L, LOGITS, temperature, g).reshape(-1)[0]) for _ in range(n)])


@test("returns_id", "sample returns one token id")
def _(L):
    check_step1(L)
    a = one_draw(L, LOGITS, 1.0, torch.Generator().manual_seed(0))
    if a.size == len(LOGITS):
        fail(f"`sample` returned {a.size} numbers for {len(LOGITS)} logits: the probabilities themselves. "
             "Draw one id from them with `torch.multinomial`.")
    if a.size != 1:
        fail(f"`sample` returned {a.size} numbers. It should return one id.")
    if a.dtype.kind != "i":
        fail(f"`sample` returned {float(a.reshape(-1)[0]):.4f}, a float. It should return the id it drew, an integer.")
    v = int(a.reshape(-1)[0])
    if not 0 <= v < len(LOGITS):
        fail(f"`sample` returned id {v}, but there are only {len(LOGITS)} logits.")


@test("uses_generator", "The same seed gives the same draws")
def _(L):
    first, second = draws(L, 1.0, seed=7, n=30), draws(L, 1.0, seed=7, n=30)
    if len(set(first.tolist())) == 1:
        fail(f"30 draws at temperature 1 from logits `{LOGITS}` all came out as id {first[0]}. "
             "That's greedy decoding. Sampling draws an id at random, in proportion to its probability.")
    if not np.array_equal(first, second):
        fail("Two runs with generators seeded the same way drew different ids. Your draw doesn't use the "
             "`generator` you were given, so nobody can reproduce your samples. Which argument of `torch.multinomial` takes it?")


@test("temperature_math", "Draws follow softmax(logits / temperature)")
def _(L):
    x = np.array(LOGITS)
    for t in (0.5, 2.0):
        got = np.bincount(draws(L, t, seed=1), minlength=len(LOGITS))[: len(LOGITS)] / DRAWS
        candidates = {
            "right": softmax(x / t),
            "times": softmax(x * t),
            "ignored": softmax(x),
        }
        dist = {k: 0.5 * np.abs(got - p).sum() for k, p in candidates.items()}
        best = min(dist, key=dist.get)
        if best == "right" and dist["right"] < 0.06:
            continue
        shown = ", ".join(f"{v:.2f}" for v in got)
        want = ", ".join(f"{v:.2f}" for v in candidates["right"])
        if best == "times":
            fail(f"At T = {t} your draws came out as [{shown}], but they should be about [{want}]. "
                 f"You multiplied by the temperature, so T = {t} {'flattened' if t < 1 else 'sharpened'} the distribution. "
                 "Low temperature should sharpen it. Divide the logits by T instead.")
        if best == "ignored":
            fail(f"At T = {t} your draws came out as [{shown}], the same as at T = 1: the temperature has no effect. "
                 "Dividing *after* softmax only rescales the probabilities, and multinomial renormalises them. "
                 "Divide the logits, before softmax.")
        fail(f"At T = {t} your draws came out as [{shown}], but softmax(logits / T) is about [{want}].")


@test("script_runs", "The script prints a sample at each temperature")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "T=1.5" not in stdout():
        fail("The loop that prints a sample at T=0.5, 1.0 and 1.5 is missing. Put it back at the bottom.")
