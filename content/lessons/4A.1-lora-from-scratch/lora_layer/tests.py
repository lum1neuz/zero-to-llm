import numpy as np
import torch
import torch.nn as nn
from harness import test, fail, warn, learner_fn, arr, LearnerError

# The lesson's layer: 6 inputs, 10 outputs. Not square, so swapped shapes can't pass by accident.
IN, OUT = 6, 10


def shape(a):
    return "[" + ", ".join(str(d) for d in a.shape) + "]"


def names(items):
    items = [f"`{i}`" for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def fresh_base(seed=0):
    torch.manual_seed(seed)
    return nn.Linear(IN, OUT)


def fresh_x(seed=5, n=3):
    return torch.randn(n, IN, generator=torch.Generator().manual_seed(seed))


def build(L, r=4, alpha=8):
    """Build the learner's layer around a fresh base, explaining placeholders and missing attributes."""
    make = learner_fn(L, "LoRALinear")
    base = fresh_base()
    layer = make(base, r, alpha)
    for n in ("A", "B", "scale"):
        if not hasattr(layer, n):
            fail(f"`self.{n}` never gets set in `__init__`. Was the line deleted or renamed?")
    holders = [f"self.{n}" for n in ("A", "B", "scale") if getattr(layer, n) is Ellipsis]
    if len(holders) == 3:
        fail("`self.A`, `self.B` and `self.scale` are still `...`, the placeholders. "
             "Start with A: r rows of small random numbers, one column per input.")
    if holders:
        what = {"self.A": "A needs r rows of small random numbers, one column per input.",
                "self.B": "B needs one row per output and r columns, and it decides where training starts.",
                "self.scale": "The update gets multiplied by it. What do alpha and r combine into?"}
        fail(f"{names(holders)} {'is' if len(holders) == 1 else 'are'} still `...`. {what[holders[0]]}")
    return base, layer


def forward(layer, x):
    """Call the learner's forward, explaining shape errors in terms of A and B."""
    try:
        out = learner_fn({"forward": layer}, "forward")(x)
    except LearnerError as e:
        if "cannot be multiplied" in str(e):
            fail(f"`forward` raised a shape error. `x` is {shape(arr(x))}, A is {shape(arr(layer.A))} and B is {shape(arr(layer.B))}. "
                 "Which of the two can `x` meet first, and does it need a `.T`?")
        raise
    if out is Ellipsis:
        fail("`forward` still returns `...`, the placeholder.")
    if arr(out) is None:
        fail(f"`forward` should return a tensor, but it returns {type(out).__name__}.")
    return out


@test("builds", "A and B are trainable parameters")
def _(L):
    _base, layer = build(L)
    for n in ("A", "B"):
        v = getattr(layer, n)
        if isinstance(v, nn.Parameter):
            if not v.requires_grad:
                fail(f"`self.{n}` is a parameter with `requires_grad=False`, so it can't train. A and B are the only things that should.")
            continue
        if isinstance(v, torch.Tensor):
            fail(f"`self.{n}` is a plain tensor, so `model.parameters()` never lists it and the optimizer would never see it. "
                 "What does a module need its trainable tensors wrapped in?")
        fail(f"`self.{n}` should be an `nn.Parameter`, but it's {type(v).__name__}.")


@test("shapes", "A is [r, in] and B is [out, r]")
def _(L):
    _base, layer = build(L, r=4, alpha=8)
    A, B = arr(layer.A), arr(layer.B)
    if A.shape != (4, IN):
        if A.shape == (IN, 4):
            fail("`A` is [6, 4], that's [in, r]. A maps 6 inputs down to r = 4 numbers, and like every `nn.Linear` weight "
                 "it's stored [out, in], so [4, 6]. PEFT checkpoints store `lora_A` the same way.")
        if A.ndim == 2 and A.shape[0] == 8:
            fail(f"`A` is {shape(A)}. 8 is alpha. The rank r = 4 sets how many rows A has.")
        fail(f"`A` has shape {shape(A)}, but with r = 4 and 6 inputs it should be [4, 6].")
    if B.shape != (OUT, 4):
        if B.shape == (4, OUT):
            fail("`B` is [4, 10], that's [r, out]. B maps the r = 4 numbers back up to 10 outputs, stored [out, in], so [10, 4].")
        if B.shape in ((OUT, IN), (IN, OUT)):
            fail(f"`B` is {shape(B)}, as big as the frozen weight itself. The whole point is that B is thin: one row per output, r = 4 columns.")
        fail(f"`B` has shape {shape(B)}, but with 10 outputs and r = 4 it should be [10, 4].")


@test("base_frozen", "Only A and B train. The base layer is frozen")
def _(L):
    _base, layer = build(L)
    extra = [n for n, p in layer.named_parameters() if p.requires_grad and n not in ("A", "B")]
    if not extra:
        return
    if extra == ["base.bias"]:
        fail("`base.bias` still requires grad. `nn.Linear` owns two parameters, weight and bias. Which call freezes everything a module owns?")
    if extra == ["base.weight"]:
        fail("`base.weight` still requires grad, so the optimizer would rewrite the pretrained weights. Freeze every parameter of `self.base`.")
    fail(f"{names(extra)} still require grad, so the optimizer would update the pretrained layer too. "
         "LoRA only works if the base stays frozen. Freeze every parameter of `self.base`.")


@test("starts_as_base", "Before training, the output equals the base layer's exactly")
def _(L):
    base, layer = build(L)
    x = fresh_x()
    out = arr(forward(layer, x))
    want = arr(base(x))
    if out.shape != want.shape:
        fail(f"`forward` returns shape {shape(out)}, but the base layer gives {shape(want)} for a batch of 3.")
    if np.array_equal(out, want):
        return
    if np.any(arr(layer.B) != 0):
        gap = float(np.abs(out - want).max())
        fail(f"B starts random, so B @ A isn't zero and your layer already differs from the pretrained one by up to {gap:.3f} "
             "before any training. Fine-tuning should start from exactly the base model. Which of the two matrices has to start at zero?")
    if np.allclose(out, 0):
        fail("Your output is all zeros: `forward` returns only the update, which is zero at the start. "
             "The frozen layer's own output has to be in there too.")
    if np.allclose(out, arr(x) @ arr(base.weight).T, atol=1e-6):
        fail("Your output is `x @ W.T` without the bias. Call the frozen layer itself, `self.base(x)`, so its bias comes along.")
    fail(f"With B all zeros the update is zero, so the output should equal `base(x)`, but it's off by up to "
         f"{float(np.abs(out - want).max()):.4f}. Check the first half of `forward`.")


@test("scaled_update", "The update is (alpha / r) · x Aᵀ Bᵀ")
def _(L):
    for r, alpha in ((4, 8), (2, 16)):
        base, layer = build(L, r=r, alpha=alpha)
        if arr(layer.A).shape != (r, IN) or arr(layer.B).shape != (OUT, r):
            fail(f"With r = {r}, A should be [{r}, 6] and B [10, {r}]. Fix \"A is [r, in] and B is [out, r]\" above first.")
        g = torch.Generator().manual_seed(11 + r)
        A = torch.randn(r, IN, generator=g)
        B = torch.randn(OUT, r, generator=g)
        with torch.no_grad():
            layer.A.copy_(A)
            layer.B.copy_(B)
        x = fresh_x()
        out = arr(forward(layer, x))
        base_out = arr(base(x)).astype(np.float64)
        upd = arr(x).astype(np.float64) @ arr(A).T @ arr(B).T
        good = alpha / r
        close = lambda s: np.allclose(out, base_out + upd * s, rtol=1e-4, atol=1e-5)
        if close(good):
            continue
        if close(1.0):
            fail(f"Your update isn't scaled. With r = {r} and alpha = {alpha} it should count {good:g}× (alpha / r). "
                 "Without the scale, changing r changes how big the update is, and the learning rate you tuned stops fitting.")
        if close(r / alpha):
            fail(f"Your scale is r / alpha = {r / alpha:g}. Flip it: LoRA multiplies the update by alpha / r = {good:g}.")
        if close(alpha):
            fail(f"Your scale is alpha = {alpha}, not alpha / r = {good:g}. Dividing by r keeps the update's size steady when you change the rank.")
        if close(1 / r):
            fail(f"Your scale is 1 / r = {1 / r:g}. alpha belongs on top: alpha / r = {good:g}.")
        if close(2.0):
            fail(f"Your scale is 2 no matter what. With r = {r} and alpha = {alpha} it should be {good:g}. Compute it from the arguments.")
        fail(f"With r = {r} and alpha = {alpha}, your output doesn't match base(x) + {good:g} · x Aᵀ Bᵀ "
             f"(off by up to {float(np.abs(out - base_out - upd * good).max()):.4f}). Check how `forward` builds the update.")


@test("grads_flow", "Training can start: B gets a gradient, the base gets none")
def _(L):
    _base, layer = build(L)
    x = fresh_x()
    target = torch.randn(3, OUT, generator=torch.Generator().manual_seed(9))
    out = forward(layer, x)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    grad = lambda p: arr(p.grad) if p.grad is not None else None
    moving = lambda gr: gr is not None and bool(np.any(gr != 0))
    gA, gB = grad(layer.A), grad(layer.B)
    if not moving(gA) and not moving(gB):
        if not np.any(arr(layer.A)) and not np.any(arr(layer.B)):
            fail("A and B both start at zero. B's gradient runs through A (it's built from x @ Aᵀ) and A's runs through B, "
                 "so both gradients are exactly zero and nothing will ever move. Which one has to start random?")
        fail("Neither A nor B got a gradient. Is the update in `forward` built from `self.A` and `self.B`?")
    if not moving(gB):
        warn("A starts at zero and B is random. That trains too, but the LoRA paper and PEFT start B at zero, and adapters you load later assume it.")
