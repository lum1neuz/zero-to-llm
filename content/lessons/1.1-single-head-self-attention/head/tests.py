import numpy as np
import torch
from harness import test, fail, warn, learner_fn, arr, script_error, stdout

rng = np.random.default_rng(3)
X = rng.standard_normal((2, 6, 12)).astype(np.float32)   # B=2, T=6, n_embd=12; the tests use head_size 5


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def attend(q, k, v):
    T, d = q.shape[-2], q.shape[-1]
    s = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)
    s = np.where(np.tril(np.ones((T, T))) == 1, s, -np.inf)
    return softmax(s) @ v


def check_step1(L):
    """Warn (don't fail) when step 1's causal_weights has drifted since it was solved."""
    fn = L.get("causal_weights")
    try:
        q, k = rng.standard_normal((2, 4, 3)).astype(np.float32)
        w = arr(fn(torch.tensor(q), torch.tensor(k)))
        ref = softmax(np.where(np.tril(np.ones((4, 4))) == 1, q @ k.T / np.sqrt(3), -np.inf))
        ok = w is not None and np.allclose(w, ref, atol=1e-5)
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: causal_weights no longer passes step 1's tests")


def make_head(L, n_embd=12, head_size=5):
    Head = learner_fn(L, "Head")
    torch.manual_seed(0)
    return Head(n_embd, head_size)


def weight(layer, name):
    """The layer's weight as NumPy, explaining placeholders and plain tensors."""
    if layer is Ellipsis:
        fail(f"`self.{name}` is still `...`, the placeholder. Make it a layer that maps n_embd features to head_size.")
    w = arr(getattr(layer, "weight", None))
    if w is None:
        fail(f"`self.{name}` should be a layer with a `.weight`, like `nn.Linear`, but it's {type(layer).__name__}.")
    return w


def run(L, x=X):
    head = make_head(L)
    out = learner_fn({"forward": head}, "forward")(torch.tensor(x))
    if out is None:
        fail("`forward` returns nothing. It has to return the mixed values, `[B, T, head_size]`.")
    if out is Ellipsis:
        fail("`forward` still returns `...`, the placeholder.")
    a = arr(out)
    if a is None:
        fail(f"`forward` should return a tensor, but it returns {type(out).__name__}.")
    return head, a.astype(np.float64)


def shaped(L, x=X):
    """run() for the tests after output_shape, which need the shape right first."""
    head, out = run(L, x)
    if out.shape != (2, 6, 5):
        fail("Fix \"forward returns [B, T, head_size]\" above first.")
    return head, out


@test("projections", "key, query and value each map n_embd → head_size, with no bias")
def _(L):
    check_step1(L)
    head = make_head(L)
    for name in ("key", "query", "value"):
        if not hasattr(head, name):
            fail(f"The head has no `self.{name}`. The tests look for layers called key, query and value.")
    ws = {name: weight(getattr(head, name), name) for name in ("key", "query", "value")}
    if head.key is head.query or head.key is head.value or head.query is head.value:
        fail("Two of key, query and value are the *same* layer, so they always produce the same vectors. "
             "Each one needs its own weights to learn its own job. Make three separate layers.")
    for name, w in ws.items():
        if w.shape == (12, 5):
            fail(f"`self.{name}.weight` is [12, 5], so it maps 5 features to 12. `nn.Linear(in, out)` takes the input size first, "
                 "and x arrives with n_embd features.")
        if w.shape != (5, 12):
            fail(f"With n_embd=12 and head_size=5, `self.{name}.weight` should be [5, 12], but it's [{', '.join(map(str, w.shape))}].")
    biased = [n for n in ("key", "query", "value") if getattr(getattr(head, n), "bias", None) is not None]
    if biased:
        n = sum(p.numel() for p in head.parameters())
        fail(f"`{biased[0]}` has a bias. This head uses `bias=False`, so it has exactly 3 × 12 × 5 = 180 parameters, and yours has {n}. "
             "You'll count parameters by hand in 1.2, so the numbers have to match.")


@test("output_shape", "forward returns [B, T, head_size]")
def _(L):
    _head, out = run(L)
    if out.shape == (2, 6, 5):
        return
    if out.shape == (2, 6, 6):
        fail("You return [2, 6, 6], the attention weights themselves. Use them to mix something, one row of weights per token.")
    if out.shape == (2, 6, 12):
        fail("You return [2, 6, 12]: you mixed `x` itself. Each token should pass along its *value* vector, of size head_size.")
    fail(f"For x of shape [2, 6, 12] and head_size 5 you return {list(out.shape)}. It should be [2, 6, 5].")


@test("no_future", "Changing a later token never changes an earlier output")
def _(L):
    _head, out = shaped(L)
    x2 = X.copy()
    x2[:, 4:] += 3.0
    _head, out2 = shaped(L, x2)
    moved = np.abs(out2[:, :4] - out[:, :4]).max()
    if moved > 1e-5:
        fail(f"Changing tokens 4 and 5 moved the outputs for tokens 0–3 by up to {moved:.3f}, so earlier tokens can see later ones. "
             "Mix the values with the weights from `causal_weights`, which hide the future.")


@test("matches_reference", "Each output is the causal-weighted average of the value vectors")
def _(L):
    head, out = shaped(L)
    Wk, Wq, Wv = (weight(getattr(head, n), n).astype(np.float64) for n in ("key", "query", "value"))
    x = X.astype(np.float64)
    k, q, v = x @ Wk.T, x @ Wq.T, x @ Wv.T
    if np.allclose(out, attend(q, k, v), atol=1e-4):
        return
    if np.allclose(out, attend(k, q, v), atol=1e-4):
        fail("Your `q` comes from `self.key` and your `k` from `self.query`. It still runs, but the names now lie, "
             "and when you load real GPT weights later each matrix would land in the wrong place. Swap them back.")
    if np.allclose(out, attend(q, k, k), atol=1e-4):
        fail("You mixed the keys. Keys decide *where* each token looks; the values are *what* gets passed along. Which one goes after the weights?")
    if np.allclose(out, attend(q, k, q), atol=1e-4):
        fail("You mixed the queries. The weights say where to look; the values are what gets passed along.")
    fail("Your output isn't softmax(q·kᵀ/√d) @ v for this head's own key, query and value layers. "
         "Check that each of k, q and v comes from its own layer.")


@test("script_runs", "The script runs and prints the T=256 timing")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "T=256" not in stdout():
        fail("The timing line didn't print. Put the `with torch.no_grad():` block back at the bottom: it's your systems metric.")
