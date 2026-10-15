import math
from harness import test, fail, learner_fn

CASES = [(1.0, 0.0), (0.5, 0.6931), (0.25, 1.3863), (0.1, 2.3026)]


def run(L, p):
    surprise = learner_fn(L, "surprise")
    got = surprise(p)
    if got is Ellipsis:
        fail("`surprise` still returns `...`, the placeholder. Return minus the log of p.")
    if got is None:
        fail("`surprise` gives back nothing. Hand the answer back with `return`.")
    return got


@test("sure_is_zero", "A certain guess (p = 1) has no surprise")
def _(L):
    got = run(L, 1.0)
    if not isinstance(got, (int, float)) or abs(got) > 1e-6:
        fail(f"surprise(1.0) gave {got!r}. A guess that was 100% sure and right shouldn't surprise anyone, so it should be 0. The log of 1 is 0.")


@test("matches_minus_log", "Less likely means more surprise: minus the natural log")
def _(L):
    for p, want in CASES:
        got = run(L, p)
        if isinstance(got, (int, float)) and abs(got - want) < 1e-3:
            continue
        if isinstance(got, (int, float)) and abs(got + want) < 1e-3:
            fail(f"surprise({p}) gave {got:.4f}, which is negative. The log of a number below 1 is negative, so put a minus in front: `-math.log(p)`.")
        if isinstance(got, (int, float)) and abs(got - (-math.log10(p))) < 1e-3:
            fail(f"surprise({p}) gave {got:.4f}. That's log10, which counts in powers of 10. This course uses the natural log, `math.log`.")
        if isinstance(got, (int, float)) and abs(got - (-math.log2(p))) < 1e-3:
            fail(f"surprise({p}) gave {got:.4f}. That's log2, which counts in powers of 2. This course uses the natural log, `math.log`.")
        if isinstance(got, (int, float)) and abs(got - (1 - p)) < 1e-6:
            fail(f"surprise({p}) gave {got:g}, which is 1 − p. That's the chance of being wrong, not the surprise. Use `-math.log(p)`: it grows much faster as p gets small.")
        fail(f"surprise({p}) should be about {want}, but you got {got!r}.")
