import numpy as np
import torch
from harness import test, fail, learner_fn, script_error, stdout

V = 8
# Different models from the script's: a seeded target and a blurred draft of it.
_rng = np.random.default_rng(11)
T = (_rng.standard_normal((V, V)) * 2).astype(np.float32)
D = (T + _rng.standard_normal((V, V)) * 1.2).astype(np.float32)
CASES = [([0], 12, 1), ([5, 2], 20, 3), ([7], 17, 4), ([1, 6, 4], 30, 2)]


def greedy(prompt, n):
    tokens = list(prompt)
    for _ in range(n):
        tokens.append(int(np.argmax(T[tokens[-1]])))
    return tokens


def spec(prompt, n_new, k, bonus=True, autoregressive=True, keep_draft=False, no_break=False, Dm=D):
    """Reference speculative greedy, plus the variants a learner might write. Returns (tokens, target passes)."""
    tokens, passes = list(prompt), 0
    for _ in range(n_new):
        if len(tokens) >= len(prompt) + n_new:
            break
        proposal = []
        for _ in range(k):
            ctx = tokens + proposal if autoregressive else tokens
            proposal.append(int(np.argmax(Dm[ctx[-1]])))
        passes += 1
        rows = T[tokens + proposal]
        n = len(tokens)
        for i in range(k + 1 if bonus else k):
            choice = int(np.argmax(rows[n - 1 + i]))
            if i < k and choice != proposal[i]:
                tokens.append(proposal[i] if keep_draft else choice)
                if no_break:
                    continue
                break
            tokens.append(choice)
            if i == k:
                break
    return tokens[:len(prompt) + n_new], passes


def run(L, prompt, n_new, k, Dm=D):
    """Run the learner's speculative_greedy against the test models, counting target and draft passes."""
    learner_fn(L, "speculative_greedy")   # explains a missing or renamed function
    fn = L["speculative_greedy"]
    G = fn.__globals__
    saved = {name: G.get(name) for name in ("target", "draft")}
    count = {"target": 0, "draft": 0}

    def target(tokens):
        count["target"] += 1
        return torch.tensor(T[list(tokens)])

    def draft(tokens):
        count["draft"] += 1
        return torch.tensor(Dm[list(tokens)])

    G["target"], G["draft"] = target, draft
    try:
        try:
            out = fn(list(prompt), n_new, k)
        except IndexError as e:
            if "out of bounds" in str(e):
                fail(f"`speculative_greedy` raised IndexError: {e}. Row j of `logits` predicts the token at position j + 1, "
                     f"so with `n = len(tokens)` the row that checks `proposal[i]` (position n + i) is n − 1 + i, "
                     "and the bonus row is the last one.")
            out = learner_fn(L, "speculative_greedy")(list(prompt), n_new, k)
        except Exception:  # noqa: BLE001 - let the harness explain it
            out = learner_fn(L, "speculative_greedy")(list(prompt), n_new, k)
    finally:
        G.update(saved)
    if out is None:
        fail("`speculative_greedy` returns nothing. Keep its last line, `return tokens[:len(prompt) + n_new]`.")
    if not isinstance(out, list):
        fail(f"`speculative_greedy` should return a list of token ids, but it returns {type(out).__name__}.")
    return [int(t) for t in out], count


@test("matches_greedy", "Speculative greedy gives exactly the target's greedy tokens")
def _(L):
    for prompt, n_new, k in CASES:
        got, count = run(L, prompt, n_new, k)
        right = greedy(prompt, n_new)
        if got == right:
            continue
        if got == list(prompt) and count["target"] == 0:
            fail("`speculative_greedy` returns just the prompt: no tokens were appended. Fill in `proposal`, the target pass "
                 "and the loop that appends the target's choices.")
        first = next((i for i in range(min(len(got), len(right))) if got[i] != right[i]), min(len(got), len(right)))
        where = f"With prompt {prompt} and k = {k}, token {first} is {got[first] if first < len(got) else 'missing'}, but greedy makes {right[first]}."
        if got == spec(prompt, n_new, k, keep_draft=True)[0]:
            fail(f"{where} At the first disagreement you kept the draft's token. The target's choice is the one greedy would make, "
                 "so append that and stop.")
        if got == spec(prompt, n_new, k, no_break=True)[0]:
            fail(f"{where} After the first disagreement you kept going. The target's later rows were computed after the draft's "
                 "wrong token, so they answer a question greedy never asks. Stop at the first disagreement.")
        if len(got) < len(right):
            fail(f"With prompt {prompt} and k = {k}, you return {len(got) - len(prompt)} new tokens instead of {n_new}. "
                 "Every round must append at least one token: the target's choice at the first disagreement, or the bonus.")
        fail(f"{where} Row j of `logits` is the target's choice for position j + 1. Append the target's choice for each "
             "drafted position until one disagrees.")


@test("one_target_pass", "Each round costs one target pass")
def _(L):
    prompt, n_new, k = [5, 2], 20, 3
    # First with a perfect draft (it is the target), so every round should keep all k guesses plus the bonus,
    # then with the blurred one.
    for Dm in (T, D):
        got, count = run(L, prompt, n_new, k, Dm)
        if got != greedy(prompt, n_new):
            fail("The tokens don't match greedy yet. Fix the test above first.")
        _tokens, right = spec(prompt, n_new, k, Dm=Dm)
        passes = count["target"]
        if passes != right:
            break
    else:
        return
    perfect = Dm is T
    which = "With a draft that always guesses right" if perfect else "With the blurred draft"
    if count["draft"] == 0:
        fail(f"Your proposals come from `target`, so every guess costs a full target pass ({passes} passes for {n_new} tokens; "
             f"plain greedy needs {n_new}). The draft is the cheap model: propose with `draft(...)`.")
    if perfect and passes == spec(prompt, n_new, k, bonus=False, Dm=Dm)[1]:
        fail(f"With a draft that always guesses right, {passes} target passes for {n_new} tokens, but {right} is possible. "
             f"When all {k} drafts agree, the last row of `logits` is already the target's next token, for free. "
             "Append it as a bonus before the next round.")
    if perfect and passes == spec(prompt, n_new, k, autoregressive=False, Dm=Dm)[1]:
        fail(f"With a draft that always guesses right, {passes} target passes for {n_new} tokens, but {right} is possible. "
             f"Your draft proposes the same token {k} times, because each guess only sees `tokens`. "
             "Each draft token must see the ones before it: `draft(tokens + proposal)`.")
    if passes > right:
        fail(f"{which}, {passes} target passes for {n_new} tokens, where {right} is possible. Score `tokens + proposal` in ONE target call "
             "per round and read every drafted position's check from its rows. Calling the target per position is plain "
             "greedy again.")
    fail(f"{which}, {passes} target passes for {n_new} tokens. Expected {right}: one per round.")


@test("script_runs", "The script runs and prints the comparison")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "got" not in L:
        fail("The lines at the bottom that compare with greedy and count target passes are missing. Put them back.")
    if not stdout().strip():
        fail("Nothing printed. Put the print lines back.")
