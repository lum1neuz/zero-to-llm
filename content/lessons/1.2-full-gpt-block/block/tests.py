import numpy as np
import torch
import torch.nn as nn
from harness import test, fail, warn, learner_fn, arr

B, T, C, NH = 2, 5, 12, 3
rng = np.random.default_rng(5)
X = (rng.standard_normal((B, T, C)) * 2 + 0.5).astype(np.float32)   # not already normalised


def layernorm(x):
    x = x.astype(np.float64)
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)


def check_step1(L):
    """Warn (don't fail) when step 1's attention has drifted: shape or causality."""
    Attn = L.get("CausalSelfAttention")
    try:
        torch.manual_seed(0)
        layer = Attn(C, NH)
        x = torch.tensor(X)
        out = arr(layer(x))
        x2 = X.copy()
        x2[:, 3:] += 3.0
        ok = out.shape == X.shape and np.abs(arr(layer(torch.tensor(x2)))[:, :3] - out[:, :3]).max() < 1e-5
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: CausalSelfAttention no longer passes step 1's tests")


def build(L):
    Block = learner_fn(L, "Block")
    torch.manual_seed(0)
    return Block(C, NH)


def forward(block, x=X):
    out = learner_fn({"forward": block}, "forward")(torch.tensor(x))
    if out is None:
        fail("`forward` returns nothing. It has to end with `return x`.")
    if out is Ellipsis:
        fail("`forward` returns `...`: `x` is still a placeholder.")
    a = arr(out)
    if a is None:
        fail(f"`forward` should return a tensor, but it returns {type(out).__name__}.")
    return a.astype(np.float64)


def shaped(L):
    block = build(L)
    out = forward(block)
    if out.shape != X.shape:
        fail("Fix \"The block keeps x's shape\" above first.")
    return block, out


def switch_off(block):
    """Zero the last layer of attention and of the MLP, so both sublayers output exactly 0."""
    with torch.no_grad():
        for lin in (block.attn.c_proj, block.mlp.c_proj):
            lin.weight.zero_()
            lin.bias.zero_()


@test("layer_norms", "ln_1 and ln_2 are two LayerNorms over n_embd features")
def _(L):
    check_step1(L)
    block = build(L)
    for name in ("ln_1", "ln_2"):
        ln = getattr(block, name, None)
        if ln is Ellipsis:
            fail(f"`self.{name}` is still `...`, the placeholder.")
        if isinstance(ln, nn.BatchNorm1d):
            fail(f"`self.{name}` is a BatchNorm, from 0.4. BatchNorm averages each feature across the *batch*, so one sequence's "
                 "output depends on the others in the batch. LayerNorm averages across each token's own features. Which one does a GPT use?")
        if not isinstance(ln, nn.LayerNorm):
            fail(f"`self.{name}` should be an `nn.LayerNorm`, but it's {type(ln).__name__}.")
        n = tuple(ln.weight.shape)
        if n != (C,):
            fail(f"`self.{name}` normalises {n[0]} features, but x has n_embd = {C}. It normalises each token's n_embd features.")
    if block.ln_1 is block.ln_2:
        fail("`ln_1` and `ln_2` are the *same* LayerNorm. Each one learns its own gain and bias for its own sublayer, so make two.")


@test("keeps_shape", "The block keeps x's shape: [B, T, n_embd]")
def _(L):
    out = forward(build(L))
    if out.shape != X.shape:
        fail(f"For x of shape [2, 5, 12] your block returns {list(out.shape)}. Blocks stack, so each one must hand back the shape it got.")


@test("residual", "With both sublayers switched off, the block passes x through unchanged")
def _(L):
    block = build(L)
    switch_off(block)
    out = forward(block)
    if out.shape == X.shape and np.allclose(out, X, atol=1e-4):
        return
    lead = "With both `c_proj` layers zeroed, attention and the MLP each output exactly 0. "
    if np.abs(out).max() < 1e-6:
        fail(lead + "Your block then returns all zeros: nothing carries x past the sublayers. "
             "Add each sublayer's output *onto* x, so x always has a direct path through.")
    if np.allclose(out, layernorm(X), atol=1e-3) or np.allclose(out, layernorm(layernorm(X)), atol=1e-3):
        fail(lead + "Your block then returns x normalised, not x. You're normalising the residual stream itself (post-norm, the 2017 layout). "
             "GPT-2 normalises only each sublayer's *input* and adds the result to x untouched.")
    fail(lead + "Your block should then return x exactly, but it changes it. Only the sublayers' outputs should be added to x.")


@test("pre_norm", "Each sublayer reads its own LayerNorm of x, in order")
def _(L):
    block, _ = shaped(L)
    with torch.no_grad():   # give the two LayerNorms different gains, so swapping them shows
        block.ln_1.weight.copy_(torch.tensor(np.linspace(0.5, 1.5, C).astype(np.float32)))
        block.ln_2.weight.copy_(torch.tensor(np.linspace(1.5, 0.5, C).astype(np.float32)))
        block.ln_2.bias.fill_(0.3)
    out = forward(block)
    f = lambda m, t: arr(m(torch.tensor(t.astype(np.float32)))).astype(np.float64)
    attn, mlp, ln1, ln2 = block.attn, block.mlp, block.ln_1, block.ln_2
    x1 = X + f(attn, f(ln1, X))
    if np.allclose(out, x1 + f(mlp, f(ln2, x1)), atol=1e-4):
        return
    candidates = (
        (X + f(attn, X), "Attention reads x raw. In a pre-norm block each sublayer reads a normalised copy: `self.attn(self.ln_1(x))`."),
        (X + f(ln1, f(attn, X)), "You normalise attention's *output*. The LayerNorm goes on the way in: `self.attn(self.ln_1(x))`."),
        (x1 + f(mlp, x1), "The MLP reads x raw. Put `ln_2` in front of it, like `ln_1` in front of attention."),
        (x1 + f(mlp, f(ln1, x1)), "Your MLP reads `ln_1`'s output. Each sublayer has its own LayerNorm with its own learned gain: the MLP gets `ln_2`."),
        (x1 + f(mlp, f(ln2, X)), "Your MLP reads the x from *before* attention, so the two sublayers run side by side (GPT-J does that on purpose). "
                                 "GPT-2 runs them in sequence: the MLP sees x after attention has added its part."),
        (X + f(attn, f(ln2, X)), "Attention reads `ln_2`. The numbers say which comes first: `ln_1` feeds attention, `ln_2` feeds the MLP."),
    )
    x_sw = X + f(attn, f(ln2, X))
    candidates += ((x_sw + f(mlp, f(ln1, x_sw)), "You swapped the LayerNorms: `ln_1` feeds attention and `ln_2` feeds the MLP."),)
    for want, story in candidates:
        if np.allclose(out, want, atol=1e-4) or np.allclose(out, want + f(mlp, f(ln2, want)), atol=1e-4):
            fail(story)
    fail("Your block doesn't compute x + attn(ln_1(x)), then x + mlp(ln_2(x)). Check what each sublayer reads.")
