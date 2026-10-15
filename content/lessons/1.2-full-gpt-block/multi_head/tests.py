import numpy as np
import torch
from harness import test, fail, learner_fn, arr, LearnerError

B, T, C, NH = 2, 6, 12, 3   # 3 heads of 4 features each
HS = C // NH
rng = np.random.default_rng(11)
X = rng.standard_normal((B, T, C)).astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def attend(q, k, v):
    t, d = q.shape[-2], q.shape[-1]
    s = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)
    s = np.where(np.tril(np.ones((t, t))) == 1, s, -np.inf)
    return softmax(s) @ v


def reference(x, layer, split="ok", merge="ok"):
    """The attention layer in NumPy, from the learner's own c_attn and c_proj weights."""
    W, b = (arr(p).astype(np.float64) for p in (layer.c_attn.weight, layer.c_attn.bias))
    Wp, bp = (arr(p).astype(np.float64) for p in (layer.c_proj.weight, layer.c_proj.bias))
    qkv = x.astype(np.float64) @ W.T + b
    q, k, v = qkv[..., :C], qkv[..., C:2 * C], qkv[..., 2 * C:]
    if split == "ok":
        heads = lambda t: t.reshape(B, T, NH, HS).transpose(0, 2, 1, 3)
    elif split == "no_transpose":
        heads = lambda t: t.reshape(B, NH, T, HS)
    else:  # "hs_nh": view(B, T, hs, n_head)
        heads = lambda t: t.reshape(B, T, HS, NH).transpose(0, 2, 1, 3)
    y = attend(heads(q), heads(k), heads(v))
    y = y.transpose(0, 2, 1, 3).reshape(B, T, C) if merge == "ok" else y.reshape(B, T, C)
    return y @ Wp.T + bp


def build(L):
    Attn = learner_fn(L, "CausalSelfAttention")
    torch.manual_seed(0)
    return Attn(C, NH)


def run(L, x=X):
    layer = build(L)
    try:
        out = learner_fn({"forward": layer}, "forward")(torch.tensor(x))
    except LearnerError as e:
        if "batch2 tensor" in str(e):
            fail("`scaled_dot_product_attention` got q, k and v in different layouts. It needs all three as "
                 f"[B, n_head, T, hs], here [2, 3, 6, 4]. Did each of them get both the view *and* the transpose?")
        raise
    a = arr(out)
    if a is None:
        fail(f"`forward` should return a tensor, but it returns {type(out).__name__}.")
    return layer, a.astype(np.float64)


def leak_story(L, layer, out):
    for split, merge, story in (
        ("no_transpose", "ok",
         "`view(B, n_head, T, hs)` cuts the [T, C] block in reading order, so head 2's \"position 0\" is really a slice of token 4. "
         "Split the features first, `view(B, T, n_head, hs)`, then swap T and the head axis with `transpose(1, 2)`."),
        ("ok", "no_transpose",
         "Reshaping [B, n_head, T, hs] straight to [B, T, C] reads head 0's tokens 0, 1 and 2 into output position 0. "
         "Swap the head axis and T back first, `transpose(1, 2)`, so each position collects its own heads."),
    ):
        if np.allclose(out, reference(X, layer, split, merge), atol=1e-4):
            return story
    return None


@test("output_shape", "forward returns [B, T, C]")
def _(L):
    _layer, out = run(L)
    if out.shape != (B, T, C):
        fail(f"For x of shape [2, 6, 12] with 3 heads, `forward` returns {list(out.shape)}. It should give back [2, 6, 12], the shape it got.")


@test("no_future", "Changing a later token never changes an earlier output")
def _(L):
    layer, out = run(L)
    x2 = X.copy()
    x2[:, 4:] += 3.0
    _layer, out2 = run(L, x2)
    moved = np.abs(out2[:, :4] - out[:, :4]).max()
    if moved <= 1e-5:
        return
    lead = f"Changing tokens 4 and 5 moved the outputs for tokens 0–3 by up to {moved:.2f}. Tokens got mixed up across positions. "
    story = leak_story(L, layer, out)
    fail(lead + (story or "Check that every reshape keeps each token's features together."))


@test("matches_reference", "Each head attends over its own slice of hs features")
def _(L):
    layer, out = run(L)
    if np.allclose(out, reference(X, layer), atol=1e-4):
        return
    if np.allclose(out, reference(X, layer, "hs_nh"), atol=1e-4):
        fail("`view(B, T, hs, n_head)` gives you hs heads of n_head features each, interleaved: head 0 gets features 0, 3, 6 and 9. "
             "Each head should own one contiguous slice of hs features, so n_head comes before hs.")
    err = np.abs(out - reference(X, layer)).max()
    fail(f"Your output is off by up to {err:.3f} from multi-head attention with these weights. "
         "Each head should take one contiguous slice of hs features from q, k and v.")
