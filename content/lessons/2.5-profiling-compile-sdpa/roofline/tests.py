from harness import test, fail, learner_fn, as_float, script_error, stdout

GPU = dict(peak_tflops=71, bandwidth_gbs=936)
# (label, params, batch, bytes per param, gpu)
PROBES = [
    ("Llama 3.1 8B in fp16 at batch 1 on the 3090", 8e9, 1, 2, GPU),
    ("Llama 3.1 8B in fp16 at batch 256 on the 3090", 8e9, 256, 2, GPU),
    ("a 70B model in int8 at batch 8 on an H100", 70e9, 8, 1, dict(peak_tflops=989, bandwidth_gbs=3350)),
    ("a 30M model in fp32 at batch 4096 on the 3090", 30e6, 4096, 4, GPU),
]


def parts(params, batch, b, gpu):
    c = 2 * params * batch / (gpu["peak_tflops"] * 1e12) * 1e3
    m = params * b / (gpu["bandwidth_gbs"] * 1e9) * 1e3
    return c, m


WRONG = {
    "summed": (lambda c, m, B: c + m,
               "You added compute time and memory time. The GPU overlaps them: while one tile computes, the next one loads. "
               "So the step takes as long as the slower of the two."),
    "min": (lambda c, m, B: min(c, m),
            "Your step takes as long as the *faster* of the two. It can't finish until the slower one is done."),
    "bytes_per_token": (lambda c, m, B: max(c, m * B),
                        "Your step reads the weights once per token in the batch. They're read once per step and shared by "
                        "every token, which is the whole point of batching (2.4)."),
    "no_batch_flops": (lambda c, m, B: max(c / B, m),
                       "Your FLOPs don't grow with the batch, so batch 256 never becomes compute-bound. Every token in the batch "
                       "needs its own 2 FLOPs per parameter."),
    "one_flop": (lambda c, m, B: max(c / 2, m),
                 "You count 1 FLOP per parameter. Each weight does a multiply and an add per token: 2 FLOPs."),
    "bits": (lambda c, m, B: max(c, m * 8),
             "You multiplied by 8. `bytes_per_param` is already in bytes (fp16 is 2), so the weights weigh params × bytes_per_param."),
}


def call(L, probe):
    fn = learner_fn(L, "decode_step_ms")
    _label, params, batch, b, gpu = probe
    out = fn(params, batch, b, **gpu)
    if out is Ellipsis:
        fail("`decode_step_ms` returns `...`, the placeholder.")
    if not isinstance(out, tuple) or len(out) != 2:
        fail("`decode_step_ms` should return two things: `return ms, bound`.")
    ms, bound = out
    if ms is Ellipsis:
        fail("`ms` is still `...`, the placeholder.")
    if as_float(ms) is None:
        fail(f"`ms` should be a number of milliseconds, but it's a {type(ms).__name__}.")
    return as_float(ms), bound


def near(a, b):
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


@test("returns_pair", "Returns a time in ms and the name of the bound")
def _(L):
    ms, bound = call(L, PROBES[0])
    if bound is Ellipsis:
        fail("`bound` is still `...`, the placeholder. It should be the string \"memory\" or \"compute\".")
    if bound not in ("memory", "compute"):
        fail(f"`bound` is {bound!r}. It should be the string \"memory\" or \"compute\".")
    if ms <= 0:
        fail(f"For {PROBES[0][0]} you got {ms:g} ms. A step takes a positive amount of time.")


@test("step_time", "The step takes as long as the slower of compute and memory")
def _(L):
    got = [call(L, p)[0] for p in PROBES]
    want = [max(*parts(*p[1:])) for p in PROBES]
    if all(near(g, w) for g, w in zip(got, want)):
        return
    for key, (fn, story) in WRONG.items():
        if all(near(g, fn(*parts(*p[1:]), p[2])) for g, p in zip(got, PROBES)):
            i = next(i for i, (g, w) in enumerate(zip(got, want)) if not near(g, w))
            fail(f"For {PROBES[i][0]} you got {got[i]:,.1f} ms instead of {want[i]:,.1f}. {story}")
    i = next(i for i, (g, w) in enumerate(zip(got, want)) if not near(g, w))
    c, m = parts(*PROBES[i][1:])
    fail(f"For {PROBES[i][0]} you got {got[i]:,.2f} ms, but compute takes {c:,.2f} ms and reading the weights {m:,.2f} ms, "
         f"so the step takes {want[i]:,.2f}. Check `flops` and `bytes_moved` first.")


@test("names_bound", "Names the slower side as the bound")
def _(L):
    for p in PROBES:
        ms, bound = call(L, p)
        c, m = parts(*p[1:])
        want = "compute" if c > m else "memory"
        if bound == want:
            continue
        fail(f"For {p[0]}, compute takes {c:,.2f} ms and memory {m:,.2f} ms, so the GPU is waiting on {want}, "
             f"but `bound` says {bound!r}. The bound is the side that takes longer.")


@test("script_runs", "The script prints four batch sizes")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if stdout().count("ms/step") < 4:
        fail("The loop that prints Llama 3.1 8B at four batch sizes is missing. Put it back at the bottom.")
