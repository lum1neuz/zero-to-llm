import math
from harness import test, fail, learner_fn

DEFAULT = dict(max_lr=6e-4, min_lr=6e-5, warmup_steps=10, max_steps=50)


def e(v):
    """6e-4 as 6.0e−4, like the lesson text."""
    if v == 0:
        return "0"
    m, ex = f"{v:.1e}".split("e")
    return f"{m}e{int(ex)}".replace("-", "−")


def reference(step, max_lr, min_lr, warmup_steps, max_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    p = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * p)) * (max_lr - min_lr)


def schedule(L, steps, config=None):
    """The learner's lr at each step, run with the given config in the script's globals."""
    fn = learner_fn(L, "get_lr")
    g = getattr(L.get("get_lr"), "__globals__", {})
    config = config or DEFAULT
    saved = {k: g[k] for k in config if k in g}
    g.update(config)
    try:
        out = []
        for s in steps:
            lr = fn(s)
            if lr is Ellipsis:
                fail(f"`get_lr({s})` returns `...`: a `return ...` placeholder is still there.")
            if lr is None:
                fail(f"`get_lr({s})` returns nothing. Every branch has to return a learning rate.")
            try:
                out.append(float(lr))
            except (TypeError, ValueError):
                fail(f"`get_lr({s})` should return a number, but it returns {type(lr).__name__}.")
        return out
    finally:
        g.update(saved)


def close(a, b):
    return abs(a - b) <= 1e-9 + 1e-4 * abs(b)


@test("warmup", "Warmup climbs in a straight line to max_lr")
def _(L):
    lrs = schedule(L, range(10))
    if all(close(v, 6e-4) for v in lrs):
        fail("Your lr is already max_lr = 6.0e−4 at step 0, so there's no warmup. Start small and climb a little every step.")
    if lrs[-1] > 0.5:
        fail(f"At step 9 your lr is {lrs[-1]:.2f}, a fraction of the way through warmup rather than a learning rate. Scale it by max_lr.")
    steps = [b - a for a, b in zip(lrs, lrs[1:])]
    if not all(close(d, steps[0]) for d in steps) or steps[0] <= 0:
        fail(f"Your warmup goes {', '.join(e(v) for v in lrs[:4])}, … It should climb by the same amount every step, a straight line.")
    if not (0.9 * 6e-4 - 1e-12 <= lrs[-1] <= 6e-4 * (1 + 1e-4)):
        fail(f"At the last warmup step (9) your lr is {e(lrs[-1])}. It should have climbed to max_lr, 6.0e−4, by then.")


@test("decay_ends", "The decay starts at max_lr and lands on min_lr")
def _(L):
    at10, at50, at60 = schedule(L, (10, 50, 60))
    if at60 <= 0:
        fail(f"After max_steps your lr is {e(at60)}, so training stops learning. Hold it at the floor, min_lr = 6.0e−5.")
    if not close(at60, 6e-5):
        fail(f"After max_steps (step 60) your lr is {e(at60)}. It should hold at min_lr, 6.0e−5.")
    if not close(at10, 6e-4):
        p = 10 / 50
        if close(at10, 6e-5 + 0.5 * (1 + math.cos(math.pi * p)) * 5.4e-4):
            fail(f"Right after warmup your lr drops from 6.0e−4 to {e(at10)} in one step: progress starts at 0.2 instead of 0. "
                 "Count progress from the end of warmup.")
        fail(f"The first step after warmup (10) has lr {e(at10)}. The decay should start from max_lr, 6.0e−4.")
    if at50 < 0:
        fail(f"At max_steps your lr is {e(at50)}, negative. cos runs from 1 down to −1, so squeeze it into 0…1 first: 0.5 × (1 + cos).")
    if not close(at50, 6e-5):
        if close(at50, 6e-5 + 0.5 * (1 + math.cos(math.pi * 0.8)) * 5.4e-4):
            fail(f"At max_steps your lr is {e(at50)}, not min_lr: progress only reaches 0.8 there. "
                 "Divide by the length of the decay, max_steps − warmup_steps, so progress ends at exactly 1.")
        if close(at50, 6e-5 + 0.5 * (1 + math.cos(1.0)) * 5.4e-4):
            fail(f"At max_steps your lr is {e(at50)}. cos(progress) only turns through 1 radian. "
                 "Multiply by π, so progress from 0 to 1 covers half a turn, from cos 0 = 1 to cos π = −1.")
        fail(f"At max_steps (step 50) your lr is {e(at50)}. The decay should land exactly on min_lr, 6.0e−5.")


@test("cosine_shape", "In between, lr follows half a cosine")
def _(L):
    lrs = schedule(L, range(10, 51))
    if any(b > a + 1e-12 for a, b in zip(lrs, lrs[1:])):
        fail("Your lr goes back up somewhere between warmup and max_steps. After warmup it should only ever fall.")
    ref = [reference(s, **DEFAULT) for s in range(10, 51)]
    if all(close(a, b) for a, b in zip(lrs, ref)):
        return
    linear = [6e-4 - (s - 10) / 40 * 5.4e-4 for s in range(10, 51)]
    if all(close(a, b) for a, b in zip(lrs, linear)):
        fail(f"Your decay is a straight line: a quarter of the way in it's at {e(lrs[10])}. A cosine falls slowly at first, "
             f"steeply in the middle and slowly again near the floor, so there it's still {e(ref[10])}.")
    k = max(range(len(ref)), key=lambda i: abs(lrs[i] - ref[i]))
    fail(f"At step {k + 10} your lr is {e(lrs[k])}, but the cosine gives {e(ref[k])}.")


@test("follows_config", "Uses max_lr, min_lr, warmup_steps and max_steps")
def _(L):
    cfg = dict(max_lr=1e-3, min_lr=1e-4, warmup_steps=20, max_steps=100)
    steps = list(range(0, 121, 5)) + [19]
    lrs = schedule(L, steps, cfg)
    for s, lr in zip(steps, lrs):
        want = reference(s, **cfg)
        if s < cfg["warmup_steps"] and close(lr, cfg["max_lr"] * s / cfg["warmup_steps"]):
            continue   # warmup from 0 at step 0 is fine too
        if not close(lr, want):
            fail(f"With max_lr = 1e−3, min_lr = 1e−4, warmup_steps = 20 and max_steps = 100, step {s} gives {e(lr)} instead of {e(want)}. "
                 "Somewhere a number is typed in where one of the four settings should be.")
