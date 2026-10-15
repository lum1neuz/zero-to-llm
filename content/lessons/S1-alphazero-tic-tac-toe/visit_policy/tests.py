from harness import test, fail, learner_fn, LearnerError, script_error, stdout


class FakeNode:
    def __init__(self, prior=1.0, n=0, w=0.0):
        self.prior, self.n, self.w, self.children = prior, n, w, {}

    def q(self):
        return self.w / self.n if self.n else 0.0


def fake_root():
    """Children on cells 0, 2, 4 and 7. Move 0 has the best Q but the fewest visits; 7 is legal but unvisited."""
    root = FakeNode()
    root.children = {0: FakeNode(0.25, 10, 8.0), 2: FakeNode(0.25, 30, 15.0), 4: FakeNode(0.25, 60, 18.0), 7: FakeNode(0.25)}
    root.n = 101  # one more than the children: the first simulation expanded the root and visited no child
    return root


VISITS = [10, 0, 30, 0, 60, 0, 0, 0, 0]


def close(a, b, tol=1e-6):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def fmt(pi):
    return "[" + ", ".join(f"{p:.2f}" for p in pi) + "]"


def policy(L, temperature):
    vp = learner_fn(L, "visit_policy")
    try:
        out = vp(fake_root(), temperature)
    except LearnerError as e:
        if temperature == 0 and "ZeroDivisionError" in str(e):
            fail("At temperature 0, `visit_policy` crashes with a division by zero: `1 / temperature`. Temperature 0 is the "
                 "limit as it shrinks: all the probability on the most visited move. Handle it before you raise to the power.")
        raise
    if out is None or out is Ellipsis:
        fail("`visit_policy` returns nothing. Is it still the `...` placeholder?")
    if isinstance(out, int):
        fail(f"At temperature {temperature:g}, `visit_policy` returns {out}, a move. It should still return 9 probabilities: "
             "1.0 on the most visited move and 0 everywhere else, so it can be a training target like any other.")
    try:
        out = [float(p) for p in out]
    except (TypeError, ValueError):
        fail(f"`visit_policy` should return a list of 9 numbers, but it returns {type(out).__name__}.")
    return out


@test("nine_cells", "Returns 9 probabilities, with 0 where there's no move")
def _(L):
    pi = policy(L, 1.0)
    if len(pi) == 4:
        fail("Your policy has 4 entries, one per child. A policy head has 9 outputs, one per cell, so the target needs "
             "a slot for every cell, with 0 where there's no legal move. Which cell would entry 2 of your list belong to?")
    if len(pi) != 9:
        fail(f"Your policy has {len(pi)} entries. It needs 9, one per cell.")
    off = [m for m in (1, 3, 5, 6, 8) if pi[m] != 0]
    if off:
        fail(f"Cells {off} have no child (they're taken), but your policy gives them weight: {fmt(pi)}.")


@test("follows_visits", "At temperature 1, each move's share is its share of the visits")
def _(L):
    pi = policy(L, 1.0)
    want = [n / 100 for n in VISITS]
    if close(pi, want):
        return
    if close(pi, [n / 101 for n in VISITS]):
        fail(f"Your policy sums to {sum(pi):.2f}, not 1. You divided by `root.n`, but the root's first simulation expanded it "
             "without visiting any child, so `root.n` is one more than the children's visits added up. "
             "What should the shares add up to?")
    if pi[0] == max(pi):
        fail(f"Your policy {fmt(pi)} puts the most weight on move 0, which has the best Q (0.8) but only 10 of the 100 visits. "
             "The target is the search's *visit* distribution: MCTS spends its visits on the moves it trusts, and a count "
             "is steadier than an average over a handful of games.")
    fail(f"Visits are 10, 30 and 60 on cells 0, 2 and 4 (100 in all), so the policy should be {fmt(want)}. Yours is {fmt(pi)}.")


@test("temperature", "Lower temperature sharpens; 0 picks the most visited move")
def _(L):
    pi = policy(L, 0.5)
    want = [n * n / 4600 for n in VISITS]
    if not close(pi, want):
        flat = [n ** 0.5 for n in VISITS]
        if close(pi, [x / sum(flat) for x in flat]):
            fail(f"At temperature 0.5 your policy is {fmt(pi)}, *flatter* than at 1. Lower temperature should sharpen it: "
                 "raise the visits to the power 1 / temperature, here 2.")
        if close(pi, [n / 100 for n in VISITS]):
            fail(f"At temperature 0.5 your policy {fmt(pi)} is the same as at 1, so temperature has no effect. "
                 "Raise the visit counts to the power 1 / temperature before you normalise.")
        fail(f"At temperature 0.5 each share goes with n², so the policy should be {fmt(want)}. Yours is {fmt(pi)}.")
    pi = policy(L, 0)
    if not close(pi, [0, 0, 0, 0, 1.0, 0, 0, 0, 0]):
        fail(f"At temperature 0 all the probability goes on the most visited move, cell 4: "
             f"`[0, 0, 0, 0, 1.0, 0, 0, 0, 0]`. Yours is {fmt(pi)}.")


@test("script_runs", "Your MCTS plays the perfect player")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "results" not in L or "play_perfect" not in L:
        fail("The games against the perfect player are missing. Put the `play_perfect` function and the loop at the bottom back.")
    if "draws vs perfect play" not in stdout():
        fail("The games ran but nothing printed the draw count. Put the last print line back.")
    if any(r not in (0, -1, 1) for r in L["results"]):
        fail(f"Some games ended with a result that isn't 1, −1 or 0: {L['results']}.")
