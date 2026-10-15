import math
from harness import test, fail, learner_fn, script_error, stdout

# A short schedule: 10 warmup steps, decay until step 110.
MAX, MIN, W, T = 3e-4, 3e-5, 10, 110
DECAY_STEPS = (10, 22, 35, 60, 85, 98, 109)


def lr(L, step):
    out = learner_fn(L, "lr_at")(step, MAX, W, T, MIN)
    if out is None:
        fail(f"`lr_at({step}, …)` returns nothing. Every branch has to return a learning rate.")
    if out is Ellipsis:
        part = "during warmup" if step < W else "after the end" if step >= T else "during the decay"
        fail(f"`lr_at` still returns `...`, the placeholder, {part} (step {step}).")
    if isinstance(out, bool) or not isinstance(out, (int, float)):
        fail(f"`lr_at` should return a number, but it returns {type(out).__name__}.")
    return float(out)


def cosine(step, start=W, end=T, period=math.pi, lo=MIN):
    p = (step - start) / (end - start)
    return lo + 0.5 * (MAX - lo) * (1 + math.cos(period * p))


def matches(got, f):
    return all(abs(g - f(s)) <= 1e-9 + 1e-6 * MAX for s, g in got.items())


def fmt(x):
    return f"{x:.2e}"


@test("warmup_ramp", "Warmup climbs in a straight line to max_lr")
def _(L):
    vals = [lr(L, s) for s in range(W)]
    step = MAX / W
    ok_from_one = all(abs(v - MAX * (s + 1) / W) < 1e-12 for s, v in enumerate(vals))
    ok_from_zero = all(abs(v - MAX * s / W) < 1e-12 for s, v in enumerate(vals))
    if ok_from_one or ok_from_zero:
        return
    if abs(vals[0] - MAX) < 1e-9 * MAX or vals[0] >= 0.9 * MAX:
        fail(f"At step 0 the learning rate is already {fmt(vals[0])}: the schedule starts at the peak. "
             f"Over the first {W} steps it should climb in a straight line, by max_lr / warmup = {fmt(step)} a step.")
    diffs = [b - a for a, b in zip(vals, vals[1:])]
    fail(f"During warmup the learning rate goes {', '.join(fmt(v) for v in vals[:4])}, … It should climb by the same amount every step, "
         f"max_lr / warmup = {fmt(step)}, and reach max_lr at the end of warmup. Your steps are {fmt(min(diffs))} to {fmt(max(diffs))}.")


@test("cosine_decay", "Then half a cosine from max_lr down to min_lr")
def _(L):
    got = {s: lr(L, s) for s in DECAY_STEPS}
    if matches(got, cosine):
        return
    mid = got[60]
    stories = [
        (lambda s: cosine(s, lo=0.0),
         f"Halfway through the decay you're at {fmt(mid)}, half of max_lr: your cosine decays to 0 instead of min_lr. "
         "The cosine part should only cover the gap between max_lr and min_lr, sitting on top of min_lr."),
        (lambda s: cosine(s, start=0),
         f"Right after warmup (step {W}) the rate is {fmt(got[W])}, already below max_lr: your progress counts from step 0, not from the end of warmup. "
         "The decay should start at max_lr when warmup ends. Count the steps since then."),
        (lambda s: cosine(s, period=1.0),
         f"At step {T - 1}, just before the end, the rate is still {fmt(got[T - 1])}, far above min_lr. `math.cos(progress)` only turns "
         "through 1 radian. Half a cosine is π radians, so multiply progress by π."),
        (lambda s: cosine(s, period=2 * math.pi),
         f"Halfway through the decay you're at {fmt(mid)} and by the end back up to {fmt(got[T - 1])}: that's a full cosine, down and back up. "
         "Half a cosine (π, not 2π) goes from max_lr down to min_lr once."),
        (lambda s: MAX - (MAX - MIN) * (s - W) / (T - W),
         f"Your decay is a straight line: a quarter of the way through it's at {fmt(got[35])}, but a cosine is still at {fmt(cosine(35))} there. "
         "A cosine stays near the peak longer, then drops faster in the middle. Use `math.cos`."),
        (lambda s: MIN + 0.5 * (MAX - MIN) * (1 - math.cos(math.pi * (s - W) / (T - W))),
         f"After warmup the rate falls to {fmt(got[W])}, then climbs to {fmt(got[T - 1])} by the end: it's upside down. "
         "At progress 0 the cosine is 1, so `1 + cos` starts at the top. Check the sign."),
    ]
    for f, text in stories:
        if matches(got, f):
            fail(text)
    if abs(got[W] - MAX) > 1e-6 * MAX:
        fail(f"When warmup ends (step {W}) the rate should be max_lr, {fmt(MAX)}, but it's {fmt(got[W])}.")
    fail(f"Halfway through the decay (step 60) the rate should be {fmt(cosine(60))}, halfway between max_lr and min_lr, but it's {fmt(mid)}.")


@test("after_end", "From total on, it stays at min_lr")
def _(L):
    for s in (T, T + 1, T + 50):
        v = lr(L, s)
        if abs(v - MIN) <= 1e-9:
            continue
        if v == 0:
            fail(f"At step {s} the rate drops to 0. Past the end it should stay at the floor, min_lr = {fmt(MIN)}.")
        fail(f"At step {s}, past the end, the rate is {fmt(v)}. It should stay at min_lr = {fmt(MIN)}.")


@test("script_runs", "The script runs and prints the schedule")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "step " not in stdout():
        fail("The loop that prints the schedule is missing. Put it back at the bottom.")
