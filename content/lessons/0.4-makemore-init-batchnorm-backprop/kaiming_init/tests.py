import math

import numpy as np
import torch
from harness import test, fail, learner_fn, arr


def weights(L, fan_in, fan_out, gain, seed=0):
    out = learner_fn(L, "linear_weights")(fan_in, fan_out, torch.Generator().manual_seed(seed), gain)
    if out is Ellipsis:
        fail("`linear_weights` still returns `...`, the placeholder.")
    a = arr(out)
    if a is None:
        fail(f"`linear_weights` should return a tensor, but it returns {type(out).__name__}.")
    return a


@test("shape", "Returns a [fan_in, fan_out] matrix")
def _(L):
    a = weights(L, 30, 200, 5 / 3)
    if a.shape == (30, 200):
        return
    if a.shape == (200, 30):
        fail("W comes out [200, 30], the transpose. The layer computes `h @ W`, with h of shape [batch, fan_in], so W has to be [fan_in, fan_out].")
    fail(f"For fan_in 30 and fan_out 200, W has shape {list(a.shape)}. It should be [30, 200].")


@test("kaiming_std", "Weights have mean 0 and std gain / √fan_in")
def _(L):
    fan_in, fan_out, gain = 400, 100, 2.0
    a = weights(L, fan_in, fan_out, gain).astype(np.float64)
    want = gain / math.sqrt(fan_in)
    std, mean = float(a.std()), float(a.mean())
    story = f"With fan_in {fan_in}, fan_out {fan_out} and gain {gain:g}, your weights have std {std:.3f}; they should have {want:.3f}. "
    if (a >= 0).all():
        fail(f"Every weight is positive (mean {mean:.3f}). `torch.rand` draws from [0, 1). Kaiming init uses `torch.randn`, a normal distribution centred on 0.")
    if abs(std - want) < 0.05 * want:
        return
    close = lambda v: abs(std - v) < 0.05 * v
    if close(1.0):
        fail(story + "Plain `randn` has std 1, so each layer's outputs grow with the square root of fan_in and tanh saturates. Scale it down.")
    if close(gain):
        fail(story + "That's the gain alone. Each output sums fan_in weighted inputs, so its spread grows like √fan_in. What undoes that?")
    if close(1 / math.sqrt(fan_in)):
        fail(story + "That's 1/√fan_in with the gain left out. tanh squashes its input, so without the gain the activations shrink a little more at every layer.")
    if close(gain / fan_in):
        fail(story + "You divided by fan_in itself. Spreads add up like variances, so the std grows with the *square root* of fan_in.")
    if close(gain / math.sqrt(fan_out)):
        fail(story + "You used fan_out. In the forward pass each output sums over the *inputs*, so the spread depends on fan_in.")
    fail(story)


@test("uses_generator", "Draws from the generator it's given")
def _(L):
    a = weights(L, 20, 10, 1.0, seed=7)
    b = weights(L, 20, 10, 1.0, seed=7)
    if not np.array_equal(a, b):
        fail("Two calls with generators seeded the same gave different weights. The draw ignores `g` and uses the global random state. "
             "Pass `generator=g`, so a seed always reproduces the same network.")
