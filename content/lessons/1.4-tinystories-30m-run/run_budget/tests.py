from harness import test, fail, learner_fn, as_float, script_error, stdout


def budget(L, n_params=1e6, tokens=1e9, peak_tflops=100, mfu=0.5):
    fn = learner_fn(L, "run_budget")
    out = fn(n_params, tokens, peak_tflops, mfu)
    if not isinstance(out, tuple) or len(out) != 2:
        fail("`run_budget` should return two values, `flops, seconds / 3600`. Keep the return line as it was.")
    flops, hours = out
    if flops is Ellipsis:
        fail("`flops` is still `...`, the placeholder.")
    f = as_float(flops)
    if f is None:
        fail(f"`flops` should be a number, but it's {type(flops).__name__}.")
    h = as_float(hours)
    if h is None:
        fail(f"The hours should be a number, but they're {type(hours).__name__}.")
    return f, h


def near(a, b):
    return abs(a - b) <= 1e-6 * abs(b)


@test("flops", "Training compute is about 6 FLOPs per parameter per token")
def _(L):
    f, _h = budget(L)
    want = 6e15
    if near(f, want):
        return
    if near(f, 2e15):
        fail("You counted 2 FLOPs per parameter per token, the forward pass alone. "
             "The backward pass costs about twice the forward (gradients for the activations *and* the weights), so 6 in all.")
    if near(f, 4e15):
        fail("You counted 4 FLOPs per parameter per token, the backward pass alone. Add the forward pass's 2.")
    if near(f, 1e15):
        fail("You counted 1 FLOP per parameter per token. Each weight does a multiply and an add per token in the forward pass (2), and the backward pass twice that.")
    fail(f"For 1M parameters and 1B tokens you get {f:.2e} FLOPs. 6 × N × D gives 6.00e+15.")


@test("hours", "GPU-hours come from the throughput you really get")
def _(L):
    f, h = budget(L)
    want = 6e15 / (100e12 * 0.5) / 3600
    if near(h, want):
        return
    if near(h, want * 0.5):
        fail(f"Your run takes {h:.4f} h, as if the GPU ran at 100% of its peak. "
             "Real training gets a fraction of it (mfu), so divide by peak × mfu.")
    if near(h, want * 0.25):
        fail("You multiplied by mfu. It's the fraction of the peak you actually get, so it makes the run *slower*: "
             "the speed you get is peak × mfu, and the time is flops divided by that.")
    if near(h, want * 1e12):
        fail("Your time is off by a factor of 10¹². TFLOPS means 10¹² FLOPs per second, so turn peak_tflops into FLOPs per second first.")
    if near(h, want / 3600):
        fail("Your answer is 3600× too small: `seconds` already holds hours, and the return line divides by 3600 again. Keep `seconds` in seconds.")
    fail(f"6e15 FLOPs at 50% of 100 TFLOPS should take {want:.4f} h, and yours says {h:.4f} h.")


@test("script_runs", "The script prints the budget for the 30M run")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "GPU-hours" not in stdout():
        fail("The print line at the bottom is gone. Put it back to see the 30M run's budget.")
