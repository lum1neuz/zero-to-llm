import numpy as np
import torch
from harness import test, fail, learner_fn, arr, script_error, stdout, LearnerError

V = 10
PROMPT = [2, 3, 4]
N_NEW = 5


class Counter:
    """A fake model that predicts "this id + 1" at every position, and records what it was fed.

    Later rows get a small bonus, so an argmax over the whole [T, V] table lands in the last row
    and gives a flat index of V or more.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, idx):
        if idx is Ellipsis:
            fail("After the first step the model was fed `...`: `idx` is still the placeholder. "
                 "Set it to the old ids with the new one on the end.")
        a = arr(idx)
        if a is None:
            fail(f"`generate` fed the model a {type(idx).__name__}, not a tensor of ids. Pass `idx`, the tensor itself.")
        a = a.reshape(-1).astype(np.int64)
        self.calls.append(a.tolist())
        T = a.shape[0]
        logits = np.full((T, V), -2.0, dtype=np.float32)
        logits[np.arange(T), (a + 1) % V] = 3.0
        logits += 0.5 * np.arange(T, dtype=np.float32)[:, None]
        return torch.tensor(logits)


def ids(values):
    return " ".join(str(int(v)) for v in values)


def run_counter(L):
    generate = learner_fn(L, "generate")
    model = Counter()
    try:
        out = generate(model, torch.tensor(PROMPT), N_NEW)
    except LearnerError as e:
        if "zero-dimensional" in str(e):
            fail("`torch.cat` refused `next_id`: it's a 0-d tensor, a single number with no dimension to join along. "
                 "Make it a 1-element tensor first, with `.view(1)` or `.unsqueeze(0)`.")
        if "shape '[1]' is invalid for input of size" in str(e) or "Tensors must have same number of dimensions" in str(e):
            fail(f"`next_id` holds more than one id: on the {len(PROMPT)}-id prompt it had one per position, "
                 "because `pick` got every row of `logits`. Give it only the last row, `logits[-1]`.")
        raise
    if out is None:
        fail("`generate` returns nothing. Keep the `return idx` at the end.")
    if out is Ellipsis:
        if not model.calls:
            fail("`generate` never called the model: `logits`, `next_id` and `idx` are still `...`, the placeholders.")
        fail("`generate` returned `...`: `idx` is still the placeholder. Set it to the old ids with the new one on the end.")
    a = arr(out)
    if a is None:
        fail(f"`generate` should return a tensor of ids, but it returns a {type(out).__name__}.")
    return a.reshape(-1) if a.ndim == 0 else a, model.calls


@test("greedy_pick", "greedy returns the id of the largest logit")
def _(L):
    greedy = learner_fn(L, "greedy")
    row = torch.tensor([0.1, 2.0, -1.0, 0.5])
    got = greedy(row)
    if got is Ellipsis:
        fail("`greedy` still returns `...`, the placeholder.")
    a = arr(got)
    if a is None:
        fail(f"`greedy` should return a tensor holding one id, but it returns a {type(got).__name__}.")
    if a.size != 1:
        fail(f"`greedy` returned {a.size} numbers for a row of 4 logits. It should return one id.")
    v = a.reshape(-1)[0]
    if a.dtype.kind == "f":
        if abs(float(v) - 2.0) < 1e-6:
            fail("On the row `[0.1, 2.0, -1.0, 0.5]`, `greedy` returned 2.0, the largest *value*. "
                 "The next token is its *position*, 1. Which method returns the index of the max?")
        fail(f"`greedy` returned {float(v):.4f}, a float. Token ids are whole numbers: the position of the largest logit.")
    if int(v) != 1:
        fail(f"On the row `[0.1, 2.0, -1.0, 0.5]`, `greedy` returned {int(v)}, but the largest logit, 2.0, sits at position 1.")


@test("returns_tokens", "generate returns the prompt plus one new id per step")
def _(L):
    a, calls = run_counter(L)
    T, want = len(PROMPT), len(PROMPT) + N_NEW
    if a.dtype.kind != "i":
        fail(f"`generate` returned {a.dtype.name} values. Token ids are integers: keep `idx` as int64 when you append.")
    if (a >= V).any() or (a < 0).any():
        bad = int(a[(a >= V) | (a < 0)][0])
        fail(f"`generate` produced the id {bad}, but this model's vocabulary has only {V} ids. "
             "An argmax over the whole `[T, V]` logits counts every entry of the table, row after row. "
             "Pick from the last row only: `logits[-1]`.")
    if a.size == want:
        if a[:T].tolist() != PROMPT:
            fail(f"The output starts {ids(a[:T])} instead of the prompt {ids(PROMPT)}. Append new ids after the prompt; don't change it.")
        return
    if a.size < T:
        fail(f"`generate` returned {a.size} id{'s' if a.size != 1 else ''} for a {T}-id prompt and {N_NEW} steps: "
             "`idx` was replaced by the new id instead of growing. Join the old ids and the new one with `torch.cat`.")
    if a.size == T:
        fail(f"`generate` returned the {T}-id prompt unchanged. Each step has to put its new id on the end of `idx`.")
    grew = [len(c) for c in calls]
    step = grew[1] - grew[0] if len(grew) > 1 else a.size - T
    if step > 1:
        fail(f"Each step added {step} ids instead of 1: `pick` got all {T} rows of logits, so it picked an id for every position. "
             "Give it only the last row, `logits[-1]`.")
    fail(f"With a {T}-id prompt and {N_NEW} steps, `generate` should return {want} ids, but it returned {a.size}.")


@test("whole_context", "Each step runs the model on the whole sequence so far")
def _(L):
    a, calls = run_counter(L)
    want = [len(PROMPT) + i for i in range(N_NEW)]
    lengths = [len(c) for c in calls]
    if lengths == want:
        return
    if lengths and all(n == 1 for n in lengths):
        fail("The model saw 1 token at each step. A transformer's next token depends on the whole context, "
             "and this loop has no memory of the earlier tokens, so run the model on all of `idx`. "
             "Keeping what it needs from old tokens is exactly what a KV cache does, in 2.2.")
    fail(f"The model was fed sequences of length {', '.join(map(str, lengths))}, but should see "
         f"{', '.join(map(str, want))}: the whole sequence so far, one longer each step.")


@test("last_position", "The next id comes from the last position's logits")
def _(L):
    a, calls = run_counter(L)
    got = a[len(PROMPT):].tolist()
    # The fake model predicts "last id + 1", so from 2 3 4 the greedy continuation counts up.
    want = [(PROMPT[-1] + 1 + i) % V for i in range(N_NEW)]
    if got == want:
        return
    if got == [(PROMPT[0] + 1) % V] * N_NEW:
        fail(f"From the prompt {ids(PROMPT)}, your loop wrote {ids(got)}. That's the prediction made at position 0, "
             "after the first token, every single step. Row t of `logits` predicts the token after position t. "
             "Which row predicts the token after the *last* one?")
    fail(f"From the prompt {ids(PROMPT)} this fake model counts up, so greedy should write {ids(want)}, "
         f"but your loop wrote {ids(got)}. Read the last row of `logits` and take its argmax.")


@test("script_runs", "Your loop writes the same 60 characters as the reference")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "out" not in L or not stdout().strip():
        fail("The lines that call `generate` and print the result are missing. Put them back at the bottom.")
    TinyLM, encode = L.get("TinyLM"), L.get("encode")
    if TinyLM is None or encode is None:
        fail("`TinyLM` or `encode` is gone from the top of the file. Put the model code back.")
    ref_model = TinyLM()
    idx = encode("the ")
    for _ in range(60):
        nxt = ref_model(idx)[-1].argmax()
        idx = torch.cat([idx, nxt.view(1)])
    if arr(L["out"]) is None or arr(L["out"]).tolist() != idx.tolist():
        fail("The tests above pass, but `out` differs from a reference greedy run on the same model. "
             "Did the last lines of the script change?")
