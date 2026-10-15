from harness import test, fail, learner_fn, as_float, script_error, stdout

CASES = [(0.75, 4), (0.5, 1), (0.9, 6), (0.3, 3)]


def tokens(a, k):
    return sum(a ** j for j in range(k + 1))


def near(x, y):
    return x is not None and abs(x - y) <= 1e-6 * max(1.0, abs(y))


def value(L, name, *args):
    out = learner_fn(L, name)(*args)
    if out is None:
        fail(f"`{name}` returns nothing. It has to `return` a number.")
    if out is Ellipsis:
        fail(f"`{name}` still returns `...`, the placeholder.")
    v = as_float(out)
    if v is None:
        fail(f"`{name}` should return a number, but it returns {type(out).__name__}.")
    return v


@test("expected_tokens", "Expected tokens per round: 1 + α + α² + … + α^k")
def _(L):
    got = {(a, k): value(L, "expected_tokens", a, k) for a, k in CASES}
    if all(near(got[c], tokens(*c)) for c in CASES):
        return
    a, k = next(c for c in CASES if not near(got[c], tokens(*c)))
    v = got[(a, k)]
    head = f"With α = {a} and k = {k} you get {v:.3f} tokens per round, but it's {tokens(a, k):.3f}."
    if all(near(got[c], tokens(*c) - 1) for c in CASES):
        fail(f"{head} You counted the drafts that survive, but every round also ends with one token from the target itself: "
             "the replacement at the first rejection, or the bonus when all k survive.")
    if all(near(got[c], tokens(c[0], c[1] - 1)) for c in CASES):
        fail(f"{head} You're one term short: the sum runs over k + 1 terms, 1 + α + … + α^k, so the closed form has "
             "α^(k + 1) on top.")
    if all(near(got[c], c[0] * c[1]) or near(got[c], c[0] * c[1] + 1) for c in CASES):
        fail(f"{head} α × k assumes every draft gets checked on its own. But a round stops at the first rejection: draft 2 "
             "only counts if draft 1 survived (α²), draft 3 if both did (α³), and so on.")
    fail(f"{head} Draft j survives only if drafts 1…j all do, with probability α^j. Add those up, plus 1 for the target's own token.")


@test("speedup", "Speedup is tokens per round over the round's cost")
def _(L):
    a, k, c = 0.75, 4, 0.15
    v = value(L, "speedup", a, k, c)
    e = tokens(a, k)
    right = e / (k * c + 1)
    if near(v, right):
        if not near(value(L, "speedup", 0.6, 2, 0.25), tokens(0.6, 2) / (2 * 0.25 + 1)):
            fail("Right for α = 0.75, k = 4, c = 0.15, but wrong for α = 0.6, k = 2, c = 0.25. Use the arguments, not fixed numbers.")
        return
    head = f"With α = {a}, k = {k}, c = {c}, your speedup is {v:.2f}×, but it's {right:.2f}×."
    if near(v, e / (k * c)):
        fail(f"{head} You count only the draft passes. Every round also runs the target once to verify, and that pass costs 1.")
    if near(v, e):
        fail(f"{head} Drafting isn't free: the round runs the draft k times, at c of a target pass each, before the target pass.")
    if near(v, e / (c + 1)):
        fail(f"{head} You charged one draft pass, but the draft runs once per drafted token: k passes of c each.")
    if near(v, e / (k + 1)):
        fail(f"{head} You charged each draft pass as a full target pass. A draft pass costs c, a fraction of one.")
    fail(f"{head} Plain decoding makes 1 token per target pass. Speculative makes `expected_tokens` per round, at a cost of k × c + 1 target passes.")


@test("script_runs", "The script runs and prints the sweep over k")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "alpha" not in L:
        fail("The loop at the bottom that sweeps k is missing. Put it back.")
    if not stdout().strip():
        fail("Nothing printed. Put the print line back.")
