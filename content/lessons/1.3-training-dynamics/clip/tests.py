import numpy as np
import torch
from harness import test, fail, learner_fn, arr, as_float, script_error, stdout

# Two "parameters" whose gradients have norms 3 and 4, so the global norm is √(9 + 16) = 5.
G1 = [2.0, -1.0, 2.0]
G2 = [[0.0, 4.0]]


def params_with(g1=G1, g2=G2):
    a = torch.zeros(3, requires_grad=True)
    b = torch.zeros(1, 2, requires_grad=True)
    a.grad = torch.tensor(g1)
    b.grad = torch.tensor(g2)
    return [a, b]


def grads(params):
    return [arr(p.grad).astype(np.float64) for p in params]


def global_norm(gs):
    return float(np.sqrt(sum((g ** 2).sum() for g in gs)))


def clip(L, params, max_norm):
    fn = learner_fn(L, "clip_grad_norm")
    out = fn(params, max_norm)
    if out is Ellipsis:
        fail("`clip_grad_norm` returns `...`: `total_norm` is still the placeholder.")
    if out is None:
        fail("`clip_grad_norm` returns nothing. It has to end with `return total_norm`.")
    n = as_float(out)
    if n is None:
        fail(f"`clip_grad_norm` should return the total norm as a number or 0-d tensor, but it returns {type(out).__name__}.")
    return n


@test("returns_norm", "Returns one norm over all the gradients, measured before clipping")
def _(L):
    n = clip(L, params_with(), 100.0)    # far above 5, so nothing should be clipped
    if abs(n - 5.0) < 1e-4:
        return
    if abs(n - 7.0) < 1e-4:
        fail("Your total is 7: you added up the two tensors' norms (3 + 4), or the |values|. Lengths don't add like that. "
             "Square every entry of every gradient, sum all of it, then take one square root: √(9 + 16) = 5.")
    if abs(n - 25.0) < 1e-4:
        fail("Your total is 25, the sum of squares. The norm is its square root.")
    if abs(n - 4.0) < 1e-4:
        fail("Your total is 4, the last tensor's norm alone. Each pass through the loop replaced the total instead of adding to it.")
    fail(f"For gradients with norms 3 and 4, your total is {n:.3f}. Laid end to end they make one vector of length √(9 + 16) = 5.")


@test("clips_to_max", "Gradients over the limit end up with total norm max_norm, same direction")
def _(L):
    params = params_with()
    n = clip(L, params, 1.0)
    gs = grads(params)
    got = global_norm(gs)
    if abs(n - 1.0) < 1e-4 and abs(got - 1.0) < 1e-4:
        fail("You return the norm *after* clipping, 1.0. Return the norm from before, 5.0: it's what you log to spot bad batches.")
    want = [np.array(G1) / 5, np.array(G2) / 5]
    if all(np.allclose(g, w, atol=1e-5) for g, w in zip(gs, want)):
        return
    if np.allclose(got, 5.0, atol=1e-4) and all(np.allclose(g, np.array(o)) for g, o in zip(gs, (G1, G2))):
        fail("Your gradients still have norm 5.00 after clipping to 1.0, so nothing changed them. Scale each `p.grad` itself, in place. "
             "Rebinding the loop variable `p` to something new doesn't touch the gradient.")
    if abs(got - 25.0) < 1e-3:
        fail("Your gradients grew to norm 25.00. The factor should shrink a norm of 5 down to 1: max_norm divided by the total.")
    if abs(got - np.sqrt(2)) < 1e-4:
        fail("Each tensor now has norm 1, so together they're 1.41, and they point somewhere new. "
             "Clip every gradient by one shared factor, computed from the global norm.")
    if abs(got - 2.0) < 1e-4:
        fail("You clamped each value to ±1, which changes the gradient's direction: [2, −1, 2] became [1, −1, 1]. "
             "Scale every gradient by one shared factor instead, so it only gets shorter.")
    fail(f"After clipping to 1.0 your gradients have total norm {got:.3f}. Every gradient should shrink by the same factor, 1/5.")


@test("small_untouched", "Gradients already under the limit are left alone")
def _(L):
    params = params_with()
    clip(L, params, 10.0)
    got = global_norm(grads(params))
    if abs(got - 5.0) < 1e-4:
        return
    fail(f"The gradients had norm 5 and the limit is 10, but afterwards their norm is {got:.2f}. "
         "Clipping only ever shrinks: cap the factor at 1.")


@test("script_runs", "The script clips the model's gradients")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "grad norm before" not in stdout():
        fail("The print line at the bottom is gone. Put it back to see the model's gradient norm before and after.")
