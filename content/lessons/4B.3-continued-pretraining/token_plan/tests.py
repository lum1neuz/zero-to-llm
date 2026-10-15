from harness import test, fail, learner_fn


def number(out, name):
    if out is None:
        fail(f"`{name}` returns nothing.")
    if out is Ellipsis:
        fail(f"`{name}` still returns `...`, the placeholder.")
    if isinstance(out, bool) or not isinstance(out, (int, float)):
        fail(f"`{name}` should give a number, but it gives {type(out).__name__}.")
    return float(out)


def close(a, b):
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


@test("same_text", "same_text_tokens counts the control's documents in extended tokens")
def _(L):
    fn = learner_fn(L, "same_text_tokens")
    for llama, fl, fe in ((10_000_000, 2.0, 1.5), (20_000_000, 3.38, 1.62)):
        got = number(fn(llama, fl, fe), "same_text_tokens")
        want = llama * fe / fl
        if close(got, want):
            continue
        m = llama / 1e6
        if close(got, llama * fl / fe):
            fail(f"{m:.0f}M Llama tokens at {fl} tokens per word come out as {got / 1e6:.1f}M extended tokens, more than before. "
                 f"The extended tokenizer spends fewer tokens per word ({fe}), so the same text should shrink. The ratio is upside down.")
        if close(got, llama):
            fail(f"You return {m:.0f}M, the same number of tokens. But the same text costs fewer tokens with the extended vocab: "
                 f"{fe} per word instead of {fl}. Count the words, then the extended tokens they become.")
        fail(f"{m:.0f}M Llama tokens at {fl} tokens per word is {llama / fl / 1e6:.2f}M words, which at {fe} tokens per word "
             f"is {want / 1e6:.2f}M extended tokens. `same_text_tokens` gives {got / 1e6:.2f}M.")


def plan(L, tokens, per_step, tps):
    out = learner_fn(L, "run_plan")(tokens, per_step, tps)
    if not isinstance(out, tuple) or len(out) != 2:
        fail("`run_plan` should return two numbers, `return steps, hours`.")
    steps, hours = out
    if steps is Ellipsis:
        fail("`steps` is still `...`, the placeholder.")
    if hours is Ellipsis:
        fail("`hours` is still `...`, the placeholder.")
    return number(steps, "run_plan"), number(hours, "run_plan"), steps


@test("steps", "run_plan counts whole optimizer steps")
def _(L):
    for tokens, per_step, tps in ((1_000_000, 8192, 1000), (20_000_000, 8192, 790)):
        steps, _h, raw = plan(L, tokens, per_step, tps)
        exact = tokens / per_step
        if steps in (tokens // per_step, -(-tokens // per_step), round(exact)) and float(steps).is_integer():
            continue
        if close(steps, exact):
            fail(f"`steps` is {steps:.2f}, with a fraction. An optimizer step is a whole batch of {per_step} tokens, so the run "
                 "takes a whole number of them. Use `//`.")
        if close(steps, tokens * per_step):
            fail(f"`steps` is {steps:,.0f}, far more steps than there are tokens. Each step eats {per_step} tokens, "
                 "so divide the tokens by it.")
        if close(steps, per_step / tokens) or close(steps, per_step // tokens):
            fail(f"`steps` is {steps:.4f}: the division is upside down. How many batches of {per_step} tokens fit into {tokens:,}?")
        fail(f"{tokens:,} tokens in steps of {per_step} is {tokens // per_step:,} steps, but `run_plan` says {raw}.")


@test("hours", "run_plan turns tokens and throughput into GPU hours")
def _(L):
    for tokens, per_step, tps in ((1_000_000, 8192, 1000), (20_000_000, 8192, 790)):
        steps, hours, _raw = plan(L, tokens, per_step, tps)
        want = tokens / tps / 3600
        if close(hours, want):
            continue
        if close(hours, tokens / tps):
            fail(f"`hours` is {hours:,.0f}: that's seconds. {tokens:,} tokens at {tps} per second take {tokens / tps:,.0f} seconds. "
                 "How many seconds are in an hour?")
        if close(hours, tokens / tps / 60):
            fail(f"`hours` is {hours:,.1f}: that's minutes. Divide by 60 once more.")
        if close(hours, steps / tps / 3600):
            fail(f"`hours` is {hours:.4f}: you divided steps by the throughput, but the throughput is tokens per second, "
                 "not steps per second. Use the tokens.")
        fail(f"{tokens:,} tokens at {tps} tokens per second is {want:.2f} GPU hours, but `run_plan` says {hours:.4f}.")
