import torch
from harness import test, fail, warn, learner_fn


def chain():
    """"." → a → b → "." with certainty, so the only possible name is "ab"."""
    P = torch.zeros((27, 27))
    P[0, 1] = 1.0
    P[1, 2] = 1.0
    P[2, 0] = 1.0
    return P


def coin():
    """"." → a or b, 50/50, then straight to "."."""
    P = torch.zeros((27, 27))
    P[0, 1] = 0.5
    P[0, 2] = 0.5
    P[1, 0] = 1.0
    P[2, 0] = 1.0
    return P


def check_step1(L):
    fn = L.get("count_bigrams")
    try:
        ok = callable(fn) and int(fn(["ab"])[0, 1]) == 1 and int(fn(["ab"]).sum()) == 3
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: count_bigrams no longer passes step 1's tests")


def rerun(L, P):
    """Run sample_name again through learner_fn, which explains the error in plain words."""
    learner_fn(L, "sample_name")(P, torch.Generator().manual_seed(0))
    fail("`sample_name` raised an error on the first try. Run your script to see the traceback.")


def sample(L, P, g):
    learner_fn(L, "sample_name")
    try:
        out = L["sample_name"](P, g)
    except KeyError as exc:
        key = exc.args[0] if exc.args else None
        if key is Ellipsis:
            fail("`ix` is still `...`, the placeholder, so `itos[ix]` has nothing to look up. Draw the next index from row `ix` of P.")
        if isinstance(key, torch.Tensor):
            fail(f"`itos[ix]` raised KeyError: {key!r}. `itos` is keyed by plain Python ints, and `ix` is a one-element tensor. "
                 "How do you turn a one-element tensor into a plain number?")
        fail(f"`sample_name` raised KeyError: {key!r}. `itos` only has keys 0 to 26.")
    except RuntimeError as exc:
        if "cannot be converted to Scalar" in str(exc):
            fail("You drew from all of P at once, one sample per row, so there are 27 numbers where you need one. "
                 "Only one row matters at each step. Which row of P should each draw use?")
        if "invalid multinomial distribution" in str(exc):
            fail("A draw used a row of P that is all zeros. In this test only the rows for \".\", \"a\" and \"b\" hold any probability, "
                 "and the walk only ever visits those. Which row of P should each draw use?")
        rerun(L, P)
    except Exception:  # noqa: BLE001 - anything else the learner's code raises
        rerun(L, P)
    if not isinstance(out, str):
        fail(f"`sample_name` should return a string, but it returns {type(out).__name__}.")
    return out


@test("returns_name", "Follows a chain where every step is certain")
def _(L):
    check_step1(L)
    out = sample(L, chain(), torch.Generator().manual_seed(1))
    if out == "ab":
        return
    if out and set(out) == {"a"}:
        fail(f"The only possible name is \"ab\", but you got \"{out[:12]}…\": the same letter over and over. "
             "Every draw comes from the same row. Which row should the next draw come from?")
    fail(f"In a table where \".\" → a → b → \".\" is certain, the only possible name is \"ab\", but you got {out!r}.")


@test("follows_probs", "Samples in proportion to P")
def _(L):
    g = torch.Generator().manual_seed(7)
    names = [sample(L, coin(), g) for _ in range(200)]
    a = names.count("a")
    if set(names) <= {"a", "b"} and 60 <= a <= 140:
        return
    if len(set(names)) == 1:
        fail(f"After \".\", a and b each have probability 0.5, but all 200 names came out {names[0]!r}. "
             "Taking the most likely letter gives the same name every time. Generating means *drawing* at random, in proportion to P.")
    fail(f"With a and b at 0.5 each, you got {a} of 200 names \"a\". Expect around 100.")


@test("uses_generator", "The same seed gives the same names")
def _(L):
    runs = []
    for _ in range(2):
        g = torch.Generator().manual_seed(42)
        runs.append([sample(L, coin(), g) for _ in range(30)])
    if runs[0] != runs[1]:
        fail("Two runs with the same seed gave different names. The draw ignores `g`, so it uses the global random state. "
             "Pass the generator into the draw, so a seed always reproduces the same names.")
