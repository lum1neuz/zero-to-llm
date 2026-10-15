import numpy as np
import torch
from harness import test, fail, learner_fn, arr, as_float, script_error, stdout

# A 5-token vocab with 4-wide rows. vocab ≠ width, so forgetting the transpose fails loudly
# instead of multiplying by the wrong thing.
_g = torch.Generator().manual_seed(11)
E_T = torch.randn(5, 4, generator=_g)
H_T = torch.randn(5, 4, generator=_g) * 2
IDS = torch.tensor([0, 3, 1, 4, 2, 0, 1, 1, 3])


def ce(logits, targets, reduction="mean"):
    z = logits - logits.max(axis=1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    nll = -lp[np.arange(len(targets)), targets]
    return nll.sum() if reduction == "sum" else nll.mean()


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def np_of(t):
    return arr(t).astype(np.float64)


def reference(E, H, ids):
    e, h, i = np_of(E), np_of(H), arr(ids)
    return ce(e[i[:-1]] @ h.T, i[1:])


def call(L, E, H, ids):
    learner_fn(L, "step0_loss")
    try:
        out = L["step0_loss"](E.clone(), H.clone(), ids.clone())
    except Exception as e:  # noqa: BLE001
        if "cannot be multiplied" in str(e):
            fail(f"`E[...] @ H` multiplies rows that are {E.shape[1]} wide by `H`, which is [{H.shape[0]}, {H.shape[1]}]: "
                 f"{e}. `nn.Linear` computes `x @ weight.T`. Each logit is a row of E against a row of H, so use `H.T`.")
        out = learner_fn(L, "step0_loss")(E.clone(), H.clone(), ids.clone())
    if out is None:
        fail("`step0_loss` returns nothing. It has to return the mean cross-entropy.")
    if out is Ellipsis:
        fail("`step0_loss` still returns `...`, the placeholder. Fill in `logits` and the return line.")
    if isinstance(out, torch.Tensor):
        if out.numel() != 1:
            fail(f"`step0_loss` returns a tensor of shape {list(out.shape)}. It should return one number, the mean loss. "
                 "Is that the logits instead of the loss?")
        out = out.item()
    v = as_float(out)
    if v is None:
        fail(f"`step0_loss` should return a number, but it returns {type(out).__name__}.")
    return v


@test("returns_loss", "step0_loss returns one number")
def _(L):
    call(L, E_T, H_T, IDS)


@test("next_token", "It's the mean loss of predicting each next token")
def _(L):
    got = call(L, E_T, H_T, IDS)
    want = reference(E_T, H_T, IDS)
    if abs(got - want) < 1e-4:
        return
    e, h, i = np_of(E_T), np_of(H_T), arr(IDS)
    n = len(i) - 1
    near = lambda x: abs(got - x) < 1e-4
    if near(ce(e[i] @ h.T, i)):
        fail(f"Your loss is {got:.3f}: the logits at each position are scored against the same token, so the model predicts each token from itself. "
             "A position should predict the token after it. Which ids are the inputs, and which are the targets?")
    if near(ce(e[i[1:]] @ h.T, i[:-1])):
        fail(f"Your loss is {got:.3f}: each position predicts the token before it. Inputs are every token except the last, `ids[:-1]`, "
             "and each one's target is the token after it, `ids[1:]`.")
    if near(ce(softmax(e[i[:-1]] @ h.T), i[1:])):
        fail(f"Your loss is {got:.3f}, squashed towards ln(vocab): `F.cross_entropy` already applies log-softmax, so a softmax before it "
             "applies it twice. Pass the raw logits.")
    if near(ce(e[i[:-1]] @ h.T, i[1:], "sum")):
        fail(f"Your loss is {got:.3f}, the sum over all {n} predictions. A sum grows with the text's length, so two tokenizers can't "
             "be compared with it. Use the mean, `F.cross_entropy`'s default.")
    if near(ce(e[i[:-1]] @ e.T, i[1:])):
        fail(f"Your loss is {got:.3f}: the logits use `E.T`, the embedding, as the LM head. Llama 3.1 8B's head is its own matrix, "
             "`H` here, so use the `H` you're given.")
    ge, gh = L.get("E"), L.get("H")
    if isinstance(ge, torch.Tensor) and isinstance(gh, torch.Tensor) and ge.ndim == 2 and gh.ndim == 2:
        try:
            if near(reference(ge, gh, IDS)):
                fail("`step0_loss` ignores the `E` and `H` it's given and scores with the script's own matrices, so it always measures the "
                     "base model. Use the arguments, or the extended matrices never get checked.")
        except IndexError:
            pass
    fail(f"On a small test model your loss is {got:.4f}, but the mean next-token cross-entropy is {want:.4f}.")


@test("script_runs", "The script runs and prints all four losses")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "E_rand" not in L:
        fail("The comparison at the bottom of the script is missing. Put it back so you can see where each start lands.")
    if len(stdout().strip().splitlines()) < 6:
        fail("Some of the print lines are missing. Put them back so you can compare the four losses.")
