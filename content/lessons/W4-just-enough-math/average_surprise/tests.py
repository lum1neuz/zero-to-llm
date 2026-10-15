import math
from harness import test, fail, learner_fn

# [0.9, 0.1] first: its last guess has a non-zero surprise, so "total gets replaced" and "nothing added" look different
CASES = [([0.9, 0.1], 1.2040), ([0.5, 0.25, 1.0], 0.6931), ([1.0, 1.0], 0.0), ([0.1], 2.3026)]


@test("loss", "It averages the surprise of every guess")
def _(L):
    avg = learner_fn(L, "average_surprise")
    for probs, want in CASES:
        got = avg(list(probs))
        if isinstance(got, (int, float)) and abs(got - want) < 1e-3:
            continue
        if got == 0 and want != 0:
            fail(f"For {probs} you got 0, so nothing was added to `total`. Inside the loop, add this guess's surprise: `total = total + surprise(p)`.")
        if isinstance(got, (int, float)) and abs(got - (-math.log(probs[-1])) / len(probs)) < 1e-6 and len(probs) > 1:
            fail(f"For {probs} you got {got:.4f}. Only the last guess counted, because `total` gets replaced each time round. Add to it instead: `total = total + ...`.")
        if isinstance(got, (int, float)) and abs(got - sum(probs) / len(probs)) < 1e-6:
            fail(f"For {probs} you got {got:.4f}, the average probability. We want the average *surprise*: add `surprise(p)`, not `p`.")
        fail(f"For {probs} the average surprise is about {want}, but you got {got!r}.")


@test("step1_still_works", "surprise still works")
def _(L):
    surprise = learner_fn(L, "surprise")
    got = surprise(0.5)
    if not isinstance(got, (int, float)) or abs(got - 0.6931) > 1e-3:
        fail(f"surprise(0.5) now gives {got!r} instead of about 0.6931. Step 1's function has changed. Check it again.")
