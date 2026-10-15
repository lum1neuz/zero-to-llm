import numpy as np
import torch
from harness import test, fail, learner_fn, arr, script_error, stdout

TEXTS = ["the cat", "a dog", "the"]


def run(L):
    left_pad, encode = learner_fn(L, "left_pad"), L.get("encode")
    if encode is None:
        fail("`encode` is gone from the top of the file. Put the model code back.")
    prompts = [encode(t) for t in TEXTS]
    out = left_pad(prompts)
    if not isinstance(out, tuple) or len(out) != 2:
        fail("`left_pad` should return two things: `return idx, mask`.")
    idx, mask = arr(out[0]), arr(out[1])
    if idx is None or mask is None:
        fail("`left_pad` should return two tensors, `idx` and `mask`.")
    return [arr(p).tolist() for p in prompts], idx, mask


def show(row):
    return "[" + ", ".join(str(int(v)) for v in row) + "]"


@test("shapes", "idx is [B, T] ids and mask is [B, T] bools")
def _(L):
    prompts, idx, mask = run(L)
    want = (len(prompts), max(map(len, prompts)))
    if idx.shape != want or mask.shape != want:
        fail(f"For prompts of lengths {[len(p) for p in prompts]}, `idx` and `mask` should both be {list(want)}, "
             f"but they are {list(idx.shape)} and {list(mask.shape)}.")
    if mask.dtype != np.bool_:
        fail(f"`mask` is {mask.dtype.name}. Keep it bool, so `~mask` and `&` mean what they say.")
    if not idx.any() and not mask.any():
        fail("`idx` is all pads and `mask` is all False: nothing was written. Fill in the two `...` lines in the loop.")


@test("left_padded", "Each row ends with its prompt, pads in front")
def _(L):
    prompts, idx, mask = run(L)
    T = idx.shape[1]
    for b, p in enumerate(prompts):
        row = idx[b].tolist()
        if row[T - len(p):] == p and all(v == 0 for v in row[: T - len(p)]):
            continue
        if row[: len(p)] == p:
            fail(f"Row {b} is {show(row)}: the prompt sits on the left and the pads trail it, so its last column is a pad. "
                 "Generation reads the prediction at the last column, and every row's new token goes in the next one. "
                 "Which side should the pads go on?")
        fail(f"Row {b} is {show(row)}, but it should be {show([0] * (T - len(p)) + p)}: pads first, then the prompt, ending at the last column.")


@test("mask_marks_real", "mask is True exactly on the prompt tokens")
def _(L):
    prompts, idx, mask = run(L)
    T = idx.shape[1]
    want = np.array([[False] * (T - len(p)) + [True] * len(p) for p in prompts])
    if np.array_equal(mask, want):
        return
    if not mask.any():
        fail("`mask` is all False. Mark the columns where each prompt sits.")
    if np.array_equal(mask, ~want):
        fail("`mask` is True on the pads and False on the prompts. It should mark the *real* tokens.")
    if np.array_equal(mask, idx != 0):
        fail(f"Row 0 is \"{TEXTS[0]}\". Its space has id 0, the same id as the pad, so a mask built from the ids calls "
             "that space padding. Mark the columns you wrote the prompt into, not the ids that look like pads.")
    b = next(i for i in range(len(prompts)) if not np.array_equal(mask[i], want[i]))
    fail(f"Row {b} of `mask` is {[bool(v) for v in mask[b]]}, but it should be True exactly where the prompt sits: "
         f"{[bool(v) for v in want[b]]}.")


@test("script_runs", "The script prints the padded batch")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if stdout().count("tensor(") < 2:
        fail("The lines that pad `prompts` and print `idx` and `mask` are missing. Put them back at the bottom.")
