import numpy as np
import torch
import torch.nn as nn
from harness import test, fail, learner_fn, arr, as_float, script_error, stdout

N = 64 * 10  # numbers in one batch's output: 64 rows × 10 outputs


def names(items):
    items = [f"`{i}`" for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


class RefLoRA(nn.Module):
    """A known-good LoRA layer, so these tests check trainable() and train_step() on their own."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.base = nn.Linear(6, 10)
        self.base.requires_grad_(False)
        g = torch.Generator().manual_seed(seed + 1)
        self.A = nn.Parameter(torch.randn(4, 6, generator=g) * 0.3)
        self.B = nn.Parameter(torch.zeros(10, 4))
        self.scale = 2.0

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def setup(seed=0):
    model = RefLoRA(seed)
    g = torch.Generator().manual_seed(seed + 7)
    X = torch.randn(64, 6, generator=g)
    delta = torch.randn(10, 2, generator=g) @ torch.randn(2, 6, generator=g) * 0.5
    with torch.no_grad():
        Y = model.base(X) + X @ delta.T
    return model, X, Y


def np_mse(model, X, Y):
    with torch.no_grad():
        return float(((arr(model(X)).astype(np.float64) - arr(Y)) ** 2).mean())


def fresh_grad_B(model, X, Y):
    """dL/dB for the mean squared error at the model's current A and B, worked out in NumPy."""
    x, y = arr(X).astype(np.float64), arr(Y).astype(np.float64)
    A, B, s = arr(model.A).astype(np.float64), arr(model.B).astype(np.float64), model.scale
    with torch.no_grad():
        out = arr(model.base(X)).astype(np.float64) + s * x @ A.T @ B.T
    dout = 2 * (out - y) / out.size
    return s * dout.T @ (x @ A.T)


class Spy:
    """Wraps a real Adam and records what B's gradient holds whenever zero_grad() or step() runs."""

    def __init__(self, model, X, Y):
        self.model, self.X, self.Y = model, X, Y
        self.opt = torch.optim.Adam([model.A, model.B], lr=0.05)
        self.events = []

    def _grad(self):
        g = self.model.B.grad
        return None if g is None else arr(g).copy()

    def zero_grad(self, set_to_none=True):
        g = self._grad()
        self.events.append(("zero", g is not None and bool(np.any(g)), None))
        self.opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.events.append(("step", self._grad(), fresh_grad_B(self.model, self.X, self.Y)))
        self.opt.step()

    def __getattr__(self, name):
        return getattr(self.opt, name)


@test("trainable_list", "trainable(model) is a list of exactly A and B")
def _(L):
    fn = learner_fn(L, "trainable")
    model, _X, _Y = setup()
    out = fn(model)
    if out is Ellipsis:
        fail("`trainable` still returns `...`, the placeholder.")
    if out is None:
        fail("`trainable` returns nothing. It has to `return` the parameters the optimizer should update.")
    if hasattr(out, "__next__"):
        fail("`trainable` returns a generator. The script uses it twice, once to count and once for Adam, and a generator is empty "
             "after the first pass, so Adam would get nothing. Turn it into a list.")
    if not isinstance(out, (list, tuple)):
        fail(f"`trainable` should return a list of parameters, but it returns {type(out).__name__}.")
    items = list(out)
    if any(isinstance(i, tuple) for i in items):
        fail("`trainable` returns (name, parameter) pairs. The optimizer wants the parameters themselves.")
    ids = {id(p) for p in items}
    frozen = [n for n, p in model.named_parameters() if id(p) in ids and not p.requires_grad]
    if frozen:
        fail(f"`trainable` also returns {names(frozen)}, which are frozen. Adam keeps two running averages for every parameter "
             "it's given. On Llama 3.1 8B that's 64 GB of optimizer state for weights that never change. Keep only the ones that require grad.")
    missing = [n for n in ("A", "B") if id(getattr(model, n)) not in ids]
    if missing:
        fail(f"`trainable` leaves out {names(missing)}, so the optimizer would never update {'it' if len(missing) == 1 else 'them'}. "
             "Which parameters still require grad?")
    if len(items) != 2:
        fail(f"`trainable` returns {len(items)} items. It should be exactly A and B, once each.")


@test("step_trains", "train_step returns the MSE and applies one fresh gradient")
def _(L):
    step = learner_fn(L, "train_step")
    model, X, Y = setup()
    spy = Spy(model, X, Y)
    want = np_mse(model, X, Y)
    out = step(model, spy, X, Y)
    v = as_float(out)
    if v is None:
        fail(f"`train_step` should return the loss as a number, but it returns {type(out).__name__}.")
    if not np.isclose(v, want, rtol=1e-4):
        with torch.no_grad():
            diff = arr(model.base(X)).astype(np.float64) - arr(Y)  # B starts at zero, so the model's output is base(X)
        if np.isclose(v, want * N, rtol=1e-4):
            fail(f"Your loss is the *sum* of squared errors over all {N} numbers, {N}× the mean. "
                 "With a sum, the gradient's size depends on the batch size, and so does the learning rate that works.")
        if np.isclose(v, np.abs(diff).mean(), rtol=1e-4):
            fail("That's the mean *absolute* error. This step fits Y with the mean *squared* error.")
        if np.isclose(v, np.sqrt(want), rtol=1e-4):
            fail("That's the square root of the MSE. Return the mean squared error itself.")
        fail(f"`train_step` returns {v:.5f}, but the mean squared error before the step is {want:.5f}.")
    steps = [e for e in spy.events if e[0] == "step"]
    if not steps:
        fail("`opt.step()` never runs, so A and B never change. The gradients are there, but nothing applies them.")
    if len(steps) > 1:
        fail(f"One `train_step` calls `opt.step()` {len(steps)} times. One step per call.")
    g1, fresh1 = steps[0][1], steps[0][2]
    if g1 is None or not np.any(g1):
        if any(e[0] == "zero" and e[1] for e in spy.events):
            fail("You clear the gradients after `backward()` and before `opt.step()`, so the step has nothing to apply and B never moves. "
                 "Where does clearing belong?")
        fail("`B.grad` is empty when `opt.step()` runs, so nothing moves. Which call fills `.grad` with the loss's gradient?")
    step(model, spy, X, Y)
    steps = [e for e in spy.events if e[0] == "step"]
    g2, fresh2 = steps[1][1], steps[1][2]
    if g2 is not None and not np.allclose(g2, fresh2, rtol=1e-3, atol=1e-6):
        if np.allclose(g2, fresh1 + fresh2, rtol=1e-3, atol=1e-6):
            fail("Gradients pile up. On the second step `B.grad` holds step 1's gradient plus its own, because nothing cleared it. "
                 "`backward()` adds into `.grad`; it never replaces it.")
        fail("On the second step, B's gradient doesn't match the loss at that point. Check the order of the lines in `train_step`.")


@test("script_runs", "Your layer fits the target and the base weights never change")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    for name in ("model", "loss", "base", "w0"):
        if name not in L:
            fail(f"`{name}` is missing from the script at the bottom. Put the training loop back.")
    if not stdout().strip():
        fail("Nothing printed. Put the print lines back so you can watch the loss.")
    if not torch.equal(L["base"].weight, L["w0"]):
        fail("`base.weight` changed during training. LoRA only trains A and B. Check that `LoRALinear` freezes the base (step 1).")
    final = as_float(L["loss"])
    if final is None or not final < 1e-3:
        fail(f"After 200 steps the loss is still {final}. The target is the base layer plus a rank-2 change, "
             "so a rank-4 adapter can fit it almost exactly. Does step 1's `LoRALinear` still pass its tests?")
