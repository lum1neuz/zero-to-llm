from harness import test, fail, learner_fn, as_float, script_error, stdout

# The script's GPU: an RTX 3090, rounded. The tests use the same numbers.
BW = 936e9
FLOPS = 71e12


def mem_ms(n, b):
    return n * b / BW * 1000


def compute_ms(n, batch):
    return 2 * n * batch / FLOPS * 1000


def near(a, b):
    return a is not None and abs(a - b) <= 1e-4 * abs(b)


def step(L, n, b, batch):
    fn = learner_fn(L, "step_ms")
    out = fn(n, b, batch)
    if out is None:
        fail("`step_ms` returns nothing. It has to end with `return` and the step time in ms.")
    if out is Ellipsis:
        fail("`step_ms` still returns `...`, the placeholder. Start with `mem_ms`.")
    v = as_float(out)
    if v is None:
        fail(f"`step_ms` should return one number of milliseconds, but it returns {type(out).__name__}.")
    return v


@test("mem_bound_at_1", "At batch 1, an 8B fp16 model takes about 17 ms a step")
def _(L):
    v = step(L, 8e9, 2, 1)
    mem, comp = mem_ms(8e9, 2), compute_ms(8e9, 1)
    if near(v, mem):
        return
    if near(v, mem / 1000):
        fail(f"Your step takes {v:.4f}. That's seconds: bytes / (bytes per second) gives seconds. Multiply by 1000 to get ms.")
    if near(v, mem * 1000):
        fail(f"Your step takes {v:.0f}, 1000× too long. Bytes / (bytes per second) already gives seconds, so one × 1000 turns it into ms.")
    if near(v, mem / 2):
        fail(f"Your step takes {v:.2f} ms, half of 17.09. That reads 8e9 *bytes*, but each fp16 parameter is 2 bytes. "
             "Where does `bytes_per_param` go?")
    if near(v, comp):
        fail(f"Your step takes {v:.2f} ms, the time for the math alone. At batch 1 the GPU finishes the math in 0.23 ms, "
             "then waits about 17 ms for the weights to stream in. Which of the two sets the pace?")
    if near(v, mem + comp):
        fail(f"Your step takes {v:.2f} ms: you added the read time and the math time. The GPU does both at the same time, "
             "streaming the next weights while it multiplies the last ones, so the step lasts as long as the *slower* of the two.")
    fail(f"For 8e9 parameters × 2 bytes at batch 1, your step takes {v:.4g} ms. Expected about {mem:.2f} ms: "
         "16e9 bytes / 936e9 bytes per second, in ms.")


@test("batch_shares_weights", "The batch shares one read of the weights")
def _(L):
    v = step(L, 8e9, 2, 8)
    mem = mem_ms(8e9, 2)
    if near(v, mem):
        return
    if near(v, 8 * mem):
        fail(f"At batch 8 your step takes {v:.2f} ms, 8× batch 1. But the 8 sequences share one read of the weights: "
             "the GPU loads each weight once and multiplies it against all 8 rows. Which of your two times grows with the batch?")
    fail(f"At batch 8 your step takes {v:.2f} ms. The math is only 1.80 ms, so the 17.09 ms weight read still sets the pace.")


@test("compute_bound_at_256", "At batch 256 the math takes longer than the weights")
def _(L):
    v = step(L, 8e9, 2, 256)
    mem, comp = mem_ms(8e9, 2), compute_ms(8e9, 256)
    if near(v, comp):
        return
    if near(v, mem):
        fail(f"At batch 256 your step still takes {v:.2f} ms, as if the math were free. 256 sequences × 2 × 8e9 FLOPs "
             f"is 4.1e15 FLOPs, {comp:.1f} ms at 71e12 FLOP/s. Does your `compute_ms` grow with the batch, and does your return look at it?")
    if near(v, comp / 2):
        fail(f"At batch 256 your step takes {v:.2f} ms, half of {comp:.2f}. Each parameter does a multiply *and* an add "
             "per token, so it's 2 FLOPs per parameter.")
    fail(f"At batch 256 your step takes {v:.2f} ms. Expected {comp:.2f} ms: 2 × 8e9 × 256 FLOPs at 71e12 FLOP/s, in ms.")


@test("tokens_per_s", "tokens_per_s counts one token per sequence per step")
def _(L):
    fn = learner_fn(L, "tokens_per_s")
    for batch in (1, 64, 256):
        out = fn(8e9, 2, batch)
        if out is Ellipsis:
            fail("`tokens_per_s` still returns `...`, the placeholder.")
        v = as_float(out)
        if v is None:
            fail(f"`tokens_per_s` should return a number, but it returns {type(out).__name__}.")
        s = step(L, 8e9, 2, batch)
        right = max(mem_ms(8e9, 2), compute_ms(8e9, batch))
        if near(v, batch * 1000 / s) or near(v, batch * 1000 / right):
            continue
        if batch > 1 and near(v, 1000 / s):
            fail(f"At batch {batch} you report {v:.0f} tok/s, the same rate as one user. Each step makes {batch} tokens, one per sequence.")
        if near(v, batch / s):
            fail(f"At batch {batch} you report {v:.4g}. That's tokens per *millisecond*, because `step_ms` is in ms. "
                 "How many ms are in a second?")
        if near(v, 1 / s):
            fail(f"At batch {batch} you report {v:.4g}: one token per step, per millisecond. Count every sequence's token, and convert ms to seconds.")
        fail(f"At batch {batch} you report {v:.4g} tok/s. With a {s:.2f} ms step that makes {batch} tokens, expected {batch * 1000 / s:.0f}.")


@test("script_runs", "The script runs and prints the table")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "batch" not in L:
        fail("The loop at the bottom that prints the table is missing. Put it back.")
    if not stdout().strip():
        fail("Nothing printed. Put the print line back in the loop so you can see where the knee is.")
