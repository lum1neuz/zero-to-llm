import statistics
from harness import test, fail, learner_fn, as_float, script_error, stdout

# A different run from the script's: the first request arrives at 0.50 s, not 0, and one request
# (index 1) waits 0.90 s, so the mean and the median wait differ. In arrival order the middle wait
# is 0.30 s, but the median is 0.25 s.
RUN = [
    {"arrive": 0.50, "first": 0.70, "done": 2.70, "tokens": 101},
    {"arrive": 0.60, "first": 1.50, "done": 3.10, "tokens": 81},
    {"arrive": 0.90, "first": 1.20, "done": 4.15, "tokens": 121},
    {"arrive": 1.40, "first": 1.65, "done": 3.30, "tokens": 65},
    {"arrive": 2.00, "first": 2.15, "done": 3.35, "tokens": 61},
]
TTFT = [r["first"] - r["arrive"] for r in RUN]
TPOT = [(r["done"] - r["first"]) / (r["tokens"] - 1) * 1000 for r in RUN]
TOKENS = sum(r["tokens"] for r in RUN)


def near(a, b, rel=1e-4):
    return a is not None and abs(a - b) <= rel * abs(b) + 1e-9


def run(L):
    summarize = learner_fn(L, "summarize")
    out = summarize([dict(r) for r in RUN])
    if out is None:
        fail("`summarize` returns nothing. Keep the `return ttft_p50, statistics.mean(tpot), tok_per_s` line.")
    if not isinstance(out, tuple) or len(out) != 3:
        fail("`summarize` should return three things: `ttft_p50`, the mean TPOT and `tok_per_s`.")
    return out


def number(value, name):
    if value is Ellipsis:
        fail(f"`{name}` is still `...`, the placeholder.")
    if isinstance(value, list):
        fail(f"`{name}` is a list, one value per request. It should be one number for the whole run.")
    v = as_float(value)
    if v is None:
        fail(f"`{name}` should be a number, but it's {type(value).__name__}.")
    return v


@test("ttft_p50", "TTFT p50 is the median wait from each request's own arrival")
def _(L):
    v = number(run(L)[0], "ttft_p50")
    right = statistics.median(TTFT)
    if near(v, right):
        return
    if near(v, statistics.median(r["first"] for r in RUN)):
        fail(f"Your p50 is {v:.2f} s. Request 2 arrived at 0.90 s and got its first token at 1.20 s, so it waited 0.30 s, "
             "but you're timing it from 0 s, the start of the run. Subtract each request's own arrival.")
    if near(v, statistics.mean(TTFT)):
        fail(f"Your p50 is {v:.2f} s. That's the mean, and one slow request (0.90 s) pulls it up. "
             "p50 is the wait that half the users beat: which statistic is that?")
    if near(v, TTFT[len(TTFT) // 2]):
        fail(f"Your p50 is {v:.2f} s, the middle of the list in arrival order. The waits are 0.20, 0.90, 0.30, 0.25 and 0.15 s. "
             "Which value has two below it and two above?")
    if near(v, statistics.median(r["done"] - r["arrive"] for r in RUN)):
        fail(f"Your p50 is {v:.2f} s, the whole request from arrival to last token. TTFT stops at the *first* token.")
    if near(v, right * 1000):
        fail(f"Your p50 is {v:.0f}: that's in ms. Keep TTFT in seconds, like the timestamps.")
    fail(f"Your p50 is {v:.4g} s. The waits (first − arrive) are 0.20, 0.90, 0.30, 0.25 and 0.15 s, so p50 should be 0.25 s.")


@test("tpot", "TPOT counts the gaps after the first token, in ms")
def _(L):
    v = number(run(L)[1], "statistics.mean(tpot)")
    right = statistics.mean(TPOT)
    if near(v, right):
        return
    by_tokens = statistics.mean((r["done"] - r["first"]) / r["tokens"] * 1000 for r in RUN)
    if near(v, by_tokens):
        fail(f"Your TPOT is {v:.2f} ms, a little low. Request 0 makes 101 tokens: the first arrives at 0.70 s, "
             "then 100 more arrive by 2.70 s. Only the tokens after the first fall inside that window.")
    if near(v, right / 1000):
        fail(f"Your TPOT is {v:.4f}, which is seconds. Multiply by 1000 for ms.")
    for denom in (0, 1):
        if near(v, statistics.mean((r["done"] - r["arrive"]) / (r["tokens"] - denom) * 1000 for r in RUN)):
            fail(f"Your TPOT is {v:.2f} ms. It counts the wait for the first token too, which TTFT already measures. "
                 "Time per output token starts at the first token.")
    fail(f"Your mean TPOT is {v:.4g} ms. Expected {right:.2f} ms: (done − first) / (tokens − 1), in ms, averaged over requests.")


@test("throughput", "Throughput is all tokens over the run's wall-clock time")
def _(L):
    v = number(run(L)[2], "tok_per_s")
    wall = max(r["done"] for r in RUN) - min(r["arrive"] for r in RUN)
    right = TOKENS / wall
    if near(v, right):
        return
    if near(v, TOKENS / max(r["done"] for r in RUN)):
        fail(f"You report {v:.1f} tok/s. You timed the run from 0 s, but its first request arrived at 0.50 s, "
             "so nothing was running before then. Start the clock at the first arrival.")
    if near(v, TOKENS / sum(r["done"] - r["arrive"] for r in RUN)):
        fail(f"You report {v:.1f} tok/s, a third of the truth. Adding up every request's time counts them one after another, "
             "but they overlap: the server works on several at once. Divide by the time the whole run took.")
    if near(v, statistics.mean(r["tokens"] / (r["done"] - r["arrive"]) for r in RUN)):
        fail(f"You report {v:.1f} tok/s: that's how fast one user sees tokens, on average. "
             "The server makes tokens for every user at the same time. Count all of them.")
    fail(f"You report {v:.4g} tok/s. Expected {right:.1f}: {TOKENS} tokens over the {wall:.2f} s from the first arrival to the last finish.")


@test("script_runs", "The script runs and prints the summary")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "tpot_ms" not in L:
        fail("The two lines at the bottom that call `summarize` and print are missing. Put them back.")
    if not stdout().strip():
        fail("Nothing printed. Put the print line back.")
