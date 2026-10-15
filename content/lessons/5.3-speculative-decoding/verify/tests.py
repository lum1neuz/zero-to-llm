import numpy as np
import torch
from harness import test, fail, learner_fn, script_error, stdout


def T(rows):
    return torch.tensor(rows)


def call(L, p, q, drafts, gen):
    """Call the learner's verify. Explain negative sampling weights here; anything else goes to the harness."""
    learner_fn(L, "verify")   # explains a missing or renamed function
    try:
        out = L["verify"](p, q, list(drafts), gen)
    except RuntimeError as e:
        if "element < 0" in str(e):
            fail("`torch.multinomial` refused your replacement weights: p − q is negative wherever the draft proposes a token more "
                 "often than the target wants it (the rejected token, for one). Clamp at 0 first: max(0, p − q).")
        out = learner_fn(L, "verify")(p, q, list(drafts), gen)
    except Exception:  # noqa: BLE001 - let the harness explain it
        out = learner_fn(L, "verify")(p, q, list(drafts), gen)
    if out is None:
        fail("`verify` returns nothing. Keep its `return out` lines.")
    if not isinstance(out, list):
        fail(f"`verify` should return a list of tokens, but it returns {type(out).__name__}.")
    return out


def tokens(out, what):
    if any(t is Ellipsis for t in out):
        fail(f"`{what}` is still `...`, the placeholder.")
    try:
        return [int(t) for t in out]
    except (TypeError, ValueError):
        fail(f"`verify` should return token ids, but it returned {out!r}. Turn a sampled tensor into an int with `int(...)`.")


def freqs(samples, n=4):
    return np.bincount(np.array(samples), minlength=n)[:n] / len(samples)


def fmt(f):
    return "[" + ", ".join(f"{v:.2f}" for v in f) + "]"


@test("keeps_agreement", "When the target agrees completely, every draft is kept, plus a bonus")
def _(L):
    # p == q, one-hot on each drafted token: every rule that keeps agreeing drafts keeps all three.
    drafts = [1, 3, 0]
    q = np.zeros((3, 4), dtype=np.float32)
    q[range(3), drafts] = 1.0
    p = np.vstack([q, [[0.1, 0.2, 0.3, 0.4]]]).astype(np.float32)
    gen = torch.Generator().manual_seed(1)
    for _ in range(30):
        out = call(L, T(p), T(q), drafts, gen)
        if len(out) == 3 and [int(t) for t in out if t is not Ellipsis] == drafts:
            fail("All 3 drafts were kept, but nothing came after them. When every draft survives, the target's last row "
                 "(it scored one more position) gives a bonus token. Sample it and append it.")
        if out and out[-1] is Ellipsis and len(out) == 4:
            fail("`bonus` is still `...`, the placeholder.")
        got = tokens(out, "fix")
        if got[:3] != drafts:
            wrong = [i for i in range(min(3, len(got))) if got[i] != drafts[i]]
            j = wrong[0] if wrong else len(got)
            fail(f"With p equal to q at every position, p(x) / q(x) is 1, so every draft must survive. Yours rejected draft {j}. "
                 "Draw `torch.rand(1, generator=gen)` and keep x when the draw is below p(x) / q(x).")
        if len(got) != 4:
            fail(f"After keeping all 3 drafts, `verify` should add exactly one bonus token, 4 in all. Yours returned {len(got)}.")


def kept_rate(L, p_row, q_row, x, n, seed):
    """Fraction of k = 1 rounds where draft x survives: kept means [x, bonus], rejected means [fix]."""
    p = T([p_row, [0.25, 0.25, 0.25, 0.25]])
    q = T([q_row])
    gen = torch.Generator().manual_seed(seed)
    kept = 0
    for _ in range(n):
        out = call(L, p, q, [x], gen)
        kept += len(out) == 2
    return kept / n


@test("accept_rule", "A draft the target likes half as much survives half the time")
def _(L):
    # Draft token 1: the target gives it 0.3, the draft 0.6, so p / q = 0.5.
    half = kept_rate(L, [0.1, 0.3, 0.4, 0.2], [0.1, 0.6, 0.2, 0.1], 1, 1500, 2)
    if abs(half - 0.5) <= 0.05:
        return
    if half > 0.95:
        # Draft token 2: the target gives it 0.6, the draft 0.3, so p / q = 2 and it always survives.
        double = kept_rate(L, [0.1, 0.1, 0.6, 0.2], [0.3, 0.3, 0.3, 0.1], 2, 600, 3)
        if abs(double - 0.5) <= 0.08:
            fail("Your rule is upside down: a draft the target likes *half* as much always survives, and one it likes twice as "
                 "much survives half the time. Keep x with probability p(x) / q(x), target over draft.")
        fail(f"Every draft survived, even one the target likes half as much as the draft (p = 0.3, q = 0.6). "
             "If `keep` is still `...`, Python treats it as True. The rule keeps x when a uniform draw is below p(x) / q(x).")
    if half < 0.05:
        fail(f"That draft never survived (p = 0.3, q = 0.6). Comparing p with q decides the same way every time, so the "
             "output drifts away from p. Draw u = torch.rand(1, generator=gen) and keep x when u < p(x) / q(x): here, half the time.")
    if abs(half - 0.3) <= 0.05:
        fail(f"That draft survived {half:.0%} of the time, which is p(x) = 0.3 on its own. The rule divides by q(x) = 0.6, "
             "the draft's own probability, so it survives half the time.")
    fail(f"A draft with p = 0.3 and q = 0.6 survived {half:.0%} of the time. It should be p / q = 50%.")


@test("replacement", "A rejected draft is replaced from max(0, p − q)")
def _(L):
    # Draft token 0 has p = 0, so it's always rejected. max(0, p − q) = [0, 0.4, 0, 0.2]: token 1 twice as often as token 3.
    p = T([[0.0, 0.5, 0.3, 0.2], [0.25, 0.25, 0.25, 0.25]])
    q = T([[0.6, 0.1, 0.3, 0.0]])
    gen = torch.Generator().manual_seed(4)
    fixes = []
    for _ in range(1200):
        out = call(L, p, q, [0], gen)
        got = tokens(out, "fix")
        if len(got) != 1:
            fail(f"Draft 0 has p = 0, so it's always rejected, and `verify` should return just its replacement. Yours returned "
                 f"{len(got)} tokens. After a rejection, return right away: the bonus is only for rounds where every draft survived.")
        fixes.append(got[0])
    f = freqs(fixes)
    right = np.array([0, 2 / 3, 0, 1 / 3])
    if np.abs(f - right).max() <= 0.05:
        return
    if f[1] == 1.0:
        fail("Your replacement is always token 1, the most likely one. That's greedy. Sample from max(0, p − q) with "
             "`torch.multinomial(..., 1, generator=gen)`.")
    if f[0] > 0.15:
        fail(f"Token 0 came back {f[0]:.0%} of the time as its own replacement, though the target gives it p = 0. "
             "That's the draft's row q. Sample from max(0, p − q): what the target wants that the draft under-proposes.")
    if f[2] > 0.15:
        fail(f"Token 2 came up {f[2]:.0%} of the time, but p and q agree on it exactly (0.3 each): the draft already proposes "
             "it as often as the target wants, so a replacement must never pick it. Sample from max(0, p − q), not from p.")
    fail(f"Replacements came out as {fmt(f)}; max(0, p − q) = [0, 0.4, 0, 0.2] gives [0.00, 0.67, 0.00, 0.33].")


@test("bonus_sampled", "The bonus token is sampled from the target's last row")
def _(L):
    p = T([[0.25, 0.25, 0.25, 0.25], [0.1, 0.0, 0.6, 0.3]])
    q = T([[0.25, 0.25, 0.25, 0.25]])
    gen = torch.Generator().manual_seed(5)
    bonus = []
    for _ in range(1200):
        got = tokens(call(L, p, q, [3], gen), "bonus")
        if len(got) != 2:
            fail(f"With p = q the draft always survives, so `verify` should return it plus one bonus token. Yours returned {len(got)}.")
        bonus.append(got[1])
    f = freqs(bonus)
    if np.abs(f - np.array([0.1, 0.0, 0.6, 0.3])).max() <= 0.05:
        return
    if f[2] == 1.0:
        fail("Your bonus is always token 2, the most likely one. That's greedy. Sample it from the target's last row with "
             "`torch.multinomial`.")
    if abs(f[1] - 0.25) <= 0.06:
        fail(f"Your bonus tokens come out as {fmt(f)}, about uniform. That's the draft's row q, or the target's first row. "
             "The bonus comes from the target's *last* row, `p[-1]`, the position after the last draft.")
    fail(f"Your bonus tokens come out as {fmt(f)}, but the target's last row is [0.10, 0.00, 0.60, 0.30].")


@test("matches_target", "With drafts from q, the output follows the target's p")
def _(L):
    p = T([[0.10, 0.50, 0.30, 0.10], [0.25, 0.25, 0.25, 0.25], [0.70, 0.10, 0.10, 0.10]])
    q = T([[0.40, 0.40, 0.10, 0.10], [0.10, 0.60, 0.20, 0.10]])
    gen = torch.Generator().manual_seed(6)
    first = []
    for _ in range(2000):
        drafts = [int(torch.multinomial(q[i], 1, generator=gen)) for i in range(2)]
        first.append(tokens(call(L, p, q, drafts, gen), "fix")[0])
    f = freqs(first)
    if np.abs(f - np.array([0.1, 0.5, 0.3, 0.1])).max() <= 0.04:
        return
    fail(f"With drafts sampled from q, your first tokens come out as {fmt(f)}, but the target says [0.10, 0.50, 0.30, 0.10]. "
         "Keeping with p / q and replacing from max(0, p − q) is what makes the two match exactly.")


@test("script_runs", "The script runs and prints the frequencies")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "counts" not in L:
        fail("The loop at the bottom that counts first tokens is missing. Put it back.")
    if not stdout().strip():
        fail("Nothing printed. Put the print line back.")
