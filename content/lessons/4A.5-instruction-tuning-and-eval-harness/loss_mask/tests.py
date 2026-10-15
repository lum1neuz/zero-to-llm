import math
import numpy as np
import torch
from harness import test, fail, learner_fn, arr, as_float, script_error

# A two-turn conversation, so a mask that only works for one assistant turn can't pass.
CHAT = [
    ("<|begin_of_text|>", None), ("<|start_header_id|>", None), ("user", None), ("<|end_header_id|>", None),
    ("Сайн", "user"), ("уу?", "user"), ("<|eot_id|>", "user"),
    ("<|start_header_id|>", None), ("assistant", None), ("<|end_header_id|>", None),
    ("Сайн,", "assistant"), ("та", "assistant"), ("сайн", "assistant"), ("уу?", "assistant"), ("<|eot_id|>", "assistant"),
    ("<|start_header_id|>", None), ("user", None), ("<|end_header_id|>", None),
    ("Баярлалаа.", "user"), ("<|eot_id|>", "user"),
    ("<|start_header_id|>", None), ("system", None), ("<|end_header_id|>", None),
    ("Товч", "system"), ("хариул.", "system"), ("<|eot_id|>", "system"),
    ("<|start_header_id|>", None), ("assistant", None), ("<|end_header_id|>", None),
    ("Зүгээр.", "assistant"), ("<|eot_id|>", "assistant"),
]


def sample():
    vocab = {}
    ids = torch.tensor([vocab.setdefault(p, len(vocab)) for p, _ in CHAT])
    roles = [r for _, r in CHAT]
    return ids, roles, vocab


def want_labels(ids, roles):
    return np.array([i if r == "assistant" else -100 for i, r in zip(arr(ids).tolist(), roles)])


def labels_of(L, ids, roles):
    before = arr(ids).copy()
    out = learner_fn(L, "build_labels")(ids, roles)
    if out is Ellipsis:
        fail("`labels` is still `...`, the placeholder.")
    if out is None:
        fail("`build_labels` returns nothing. Keep `return labels` at the end.")
    a = arr(out)
    if a is None:
        fail(f"`build_labels` should return a tensor, but it returns {type(out).__name__}.")
    if not np.array_equal(arr(ids), before):
        fail("`ids` changed too. `labels = ids` makes a second name for the same tensor, so writing -100 into labels "
             "writes it into the model's input as well. Make a copy first.")
    return a


@test("labels_masked", "Only the assistant's tokens keep their ids. Everything else is -100")
def _(L):
    ids, roles, vocab = sample()
    got = labels_of(L, ids, roles)
    good = want_labels(ids, roles)
    if got.shape != good.shape:
        fail(f"`labels` has shape {list(got.shape)}, but there are {len(good)} tokens. One label per token.")
    if np.array_equal(got, good):
        return
    pieces = [p for p, _ in CHAT]
    wrong = [i for i in range(len(good)) if got[i] != good[i]]
    if all(got[i] == -100 for i in range(len(good)) if roles[i] == "assistant") and all(got[i] != -100 for i in range(len(good)) if roles[i] != "assistant"):
        fail("Your mask is the wrong way round: the assistant's tokens are -100 and everything else keeps its id. "
             "The model should learn to write the answers, not the questions.")
    if all(got[i] == 0 for i in wrong) and all(good[i] == -100 for i in wrong):
        fail("You fill the masked positions with 0. But 0 is a real token id here, `<|begin_of_text|>`, so the loss "
             "trains the model to predict it after every prompt word. The loss skips -100.")
    trained = [i for i in wrong if good[i] == -100 and got[i] != -100]
    if trained and all(roles[i] is None for i in trained):
        fail(f"The template's own tokens keep their ids: {', '.join(sorted({pieces[i] for i in trained}))}. "
             "The model never has to write those, the template adds them. Mask everything that isn't the assistant's.")
    if trained and any(roles[i] == "system" for i in trained):
        fail("The system message keeps its ids, so the model learns to write system prompts. Mask everything that isn't the assistant's.")
    if trained:
        i = trained[0]
        fail(f"\"{pieces[i]}\" ({roles[i] or 'template'}) keeps its id, but only the assistant's tokens should.")
    i = wrong[0]
    fail(f"\"{pieces[i]}\" said by the {roles[i]} should keep its id {int(good[i])}, but its label is {int(got[i])}.")


def ref_loss(logits, labels):
    la, lb = arr(logits).astype(np.float64)[:-1], labels[1:]
    keep = lb != -100
    m = la.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(la - m).sum(axis=1, keepdims=True)))[:, 0]
    nll = lse - la[np.arange(len(lb)), np.where(keep, lb, 0)]
    return float(nll[keep].mean()), float(nll[keep].sum()), int(keep.sum())


@test("loss_value", "sft_loss is the mean next-token loss over the assistant's tokens")
def _(L):
    ids, roles, vocab = sample()
    labels = want_labels(ids, roles)
    logits = torch.randn(len(labels), len(vocab), generator=torch.Generator().manual_seed(3))
    out = learner_fn(L, "sft_loss")(logits, torch.tensor(labels.tolist()))
    if out is Ellipsis:
        fail("`sft_loss` still returns `...`, the placeholder.")
    v = as_float(out)
    if v is None:
        fail(f"`sft_loss` should return a single number (a 0-d tensor), but it returns {type(out).__name__}.")
    mean, total, n = ref_loss(logits, labels)
    if math.isclose(v, mean, rel_tol=1e-4):
        return
    la = arr(logits).astype(np.float64)
    keep = labels != -100
    m = la.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(la - m).sum(axis=1, keepdims=True)))[:, 0]
    unshifted = float((lse - la[np.arange(len(labels)), np.where(keep, labels, 0)])[keep].mean())
    if math.isclose(v, unshifted, rel_tol=1e-4):
        fail("You score `logits[t]` against `labels[t]`, the token the model was just given. Position t predicts the *next* "
             "token, so compare `logits[:-1]` with `labels[1:]`.")
    if math.isclose(v, total, rel_tol=1e-4):
        fail(f"You return the sum over the {n} assistant tokens. Use the mean, so a long answer doesn't count as a bigger step than a short one.")
    if math.isclose(v, total / (len(labels) - 1), rel_tol=1e-4):
        fail(f"You divide by all {len(labels) - 1} positions, but only {n} are trained. A long prompt would shrink the loss "
             "without the answer getting any better. Average over the positions that count.")
    fail(f"The mean next-token loss over the {n} assistant tokens is {mean:.4f}. Yours is {v:.4f}.")


@test("script_runs", "The script prints every token's label and the loss")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "labels" not in L or "logits" not in L:
        fail("The lines that print the labels and the loss are missing. Put them back.")
