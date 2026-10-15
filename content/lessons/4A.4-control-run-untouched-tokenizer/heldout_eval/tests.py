import math
import numpy as np
import torch
from harness import test, fail, learner_fn, arr, as_float, script_error, stdout

V = 16


def sample(n, seed, nll=1.2):
    """A document of n tokens (BOS first) and logits that favour each next token."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, V, (n,), generator=g)
    ids[0] = 0
    logits = torch.randn(n, V, generator=g)
    p = math.exp(-nll)
    logits[torch.arange(n - 1), ids[1:]] += math.log((V - 1) * p / (1 - p))
    return logits, ids


def ref_nll(logits, ids, shift=True):
    la, ia = arr(logits).astype(np.float64), arr(ids)
    if shift:
        la, ia = la[:-1], ia[1:]
    elif shift is None:  # backwards: logits[1:] scored against ids[:-1]
        la, ia = la[1:], ia[:-1]
    m = la.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(la - m).sum(axis=1, keepdims=True)))[:, 0]
    return float((lse - la[np.arange(len(ia)), ia]).sum())


# Three held-out "documents" of very different lengths, so pooling and averaging disagree.
TEXTS = ["Цас орно.", "Малчид өвөлжөөндөө бэлтгэлээ базааж байна.",
         "Наадмын баяр долдугаар сарын арван нэгнээс эхэлж, гурван өдөр үргэлжилнэ. Улаанбаатар хотод цас орж байна."]
DOCS = [(t, *sample(n, 40 + i, nll)) for i, (t, n, nll) in enumerate(zip(TEXTS, (3, 18, 45), (0.4, 1.0, 1.6)))]


def nll_value(L, logits, ids):
    out = learner_fn(L, "doc_nll")(logits, ids)
    if out is Ellipsis:
        fail("`doc_nll` still returns `...`, the placeholder.")
    if out is None:
        fail("`doc_nll` returns nothing. It needs a `return`.")
    v = as_float(out)
    if v is None:
        if isinstance(out, torch.Tensor):
            fail(f"`doc_nll` returns a tensor of shape {list(out.shape)}. Sum it to one number: `reduction=\"sum\"`.")
        fail(f"`doc_nll` should return a number, but it returns {type(out).__name__}.")
    return v


@test("doc_nll", "doc_nll sums the loss of predicting each next token")
def _(L):
    logits, ids = sample(9, 7)
    v = nll_value(L, logits, ids)
    good = ref_nll(logits, ids)
    if math.isclose(v, good, rel_tol=1e-4):
        return
    if math.isclose(v, ref_nll(logits, ids, shift=False), rel_tol=1e-4):
        fail("You score `logits[t]` against `ids[t]`, the token position t was given as input. Position t predicts the *next* token, "
             "so drop the last row of logits and the first id.")
    if math.isclose(v, ref_nll(logits, ids, shift=None), rel_tol=1e-4):
        fail("Backwards: you score `logits[1:]` against `ids[:-1]`, asking each position for the token *before* it. "
             "Position t predicts ids[t + 1]: `logits[:-1]` against `ids[1:]`.")
    if math.isclose(v, good / 8, rel_tol=1e-4):
        fail("You return the *average* loss per token. `heldout` pools many documents, and a 10-token document shouldn't count "
             "as much as a 1000-token one. Return the sum, and divide once, at the end.")
    if math.isclose(v, -good, rel_tol=1e-4):
        fail("Your number is negative: that's the log-likelihood. The *negative* log-likelihood is positive, and smaller is better.")
    if math.isclose(v, good / math.log(2), rel_tol=1e-4):
        fail("Your loss is in bits. Keep it in nats (natural log), like cross_entropy gives it. `heldout` converts to bits itself.")
    fail(f"For 9 tokens (8 predictions) the summed next-token loss is {good:.4f} nats. Yours is {v:.4f}.")


def run_heldout(L, docs):
    out = learner_fn(L, "heldout")(docs)
    if not isinstance(out, tuple) or len(out) != 2:
        fail("`heldout` should return two numbers, `return ppl, bpc`.")
    ppl, bpc = out
    for name, val in (("ppl", ppl), ("bpc", bpc)):
        if val is Ellipsis:
            fail(f"`{name}` is still `...`, the placeholder.")
    ppl, bpc = as_float(ppl), as_float(bpc)
    if ppl is None or bpc is None:
        fail("`ppl` and `bpc` should both be numbers.")
    return ppl, bpc


@test("pooled_ppl", "Perplexity pools every predicted token across documents")
def _(L):
    ppl, _bpc = run_heldout(L, DOCS)
    nlls = [ref_nll(lg, ids) for _, lg, ids in DOCS]
    counts = [len(ids) - 1 for _, _, ids in DOCS]
    good = math.exp(sum(nlls) / sum(counts))
    if math.isclose(ppl, good, rel_tol=1e-4):
        return
    if math.isclose(ppl, sum(math.exp(a / n) for a, n in zip(nlls, counts)) / 3, rel_tol=1e-4):
        fail(f"You average the three documents' perplexities, so the 2-prediction document counts as much as the 44-prediction one. "
             f"Pool first: total loss over total predicted tokens, then exp. That gives {good:.3f}, not {ppl:.3f}.")
    if math.isclose(ppl, math.exp(sum(a / n for a, n in zip(nlls, counts)) / 3), rel_tol=1e-4):
        fail("You average each document's mean loss, which weights a 2-token document like a 44-token one. "
             "Add up all the loss and all the predicted tokens first, then divide once.")
    if math.isclose(ppl, math.exp(sum(nlls) / (sum(counts) + 3)), rel_tol=1e-4):
        fail("You count every token, BOS included. The first token of each document is never predicted, so each one adds "
             "`len(ids) - 1` predictions, not `len(ids)`.")
    if math.isclose(ppl, sum(nlls) / sum(counts), rel_tol=1e-4):
        fail("That's the mean loss per token, in nats. Perplexity is e to that power.")
    if math.isclose(ppl, 2 ** (sum(nlls) / sum(counts)), rel_tol=1e-4):
        fail("You raise 2 to the mean loss, but the loss is in nats (natural log). Use e, `math.exp`.")
    fail(f"Pooled over the three documents, perplexity should be {good:.3f}. Yours is {ppl:.3f}.")


@test("bits_per_char", "Bits per character turns nats into bits and divides by characters")
def _(L):
    _ppl, bpc = run_heldout(L, DOCS)
    total = sum(ref_nll(lg, ids) for _, lg, ids in DOCS)
    chars = sum(len(t) for t in TEXTS)
    good = total / math.log(2) / chars
    if math.isclose(bpc, good, rel_tol=1e-4):
        return
    if math.isclose(bpc, total / chars, rel_tol=1e-4):
        fail("That's nats per character. Divide by ln 2 to turn nats into bits.")
    tokens = sum(len(ids) - 1 for _, _, ids in DOCS)
    if math.isclose(bpc, total / math.log(2) / tokens, rel_tol=1e-4):
        fail("That's bits per *token*. Tokens are exactly what changes when 4B swaps the tokenizer. Divide by the number of characters, which doesn't.")
    if math.isclose(bpc, total / math.log(2) / sum(len(t.encode("utf-8")) for t in TEXTS), rel_tol=1e-4):
        fail("That's bits per *byte*. It's tokenizer-free too, but every Cyrillic letter is 2 bytes in UTF-8, so it comes out about half. "
             "Count characters, `len(text)`.")
    if math.isclose(bpc, total * math.log(2) / chars, rel_tol=1e-4):
        fail("You multiply by ln 2. One bit is ln 2 ≈ 0.693 nats, so nats become bits by *dividing* by ln 2.")
    fail(f"Bits per character should be {good:.4f}. Yours is {bpc:.4f}.")


@test("script_runs", "Both runs print a perplexity and bits per character")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if stdout().count("bits/char") < 2:
        fail("The loop at the bottom should print one line for each run. Put it back.")
