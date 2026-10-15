import math

# The control run (4A.4), from its log.
CONTROL_TOKENS = 20_000_000   # the 20M-token sample, in Llama 3.1 tokens (seed 0, ids saved)
CONTROL_TOK_PER_S = 790       # its throughput on a 12 GB RTX 3060
TOKENS_PER_STEP = 4 * 2048    # its batch: 4 sequences of 2,048 tokens. Keep it.

# Your extended run.
FERT_LLAMA = 3.38             # tokens per Mongolian word, Llama 3.1's tokenizer (your 1.6 audit; reference 3.38)
FERT_EXT = 1.62               # tokens per word, your extended tokenizer (4B.1)
TOK_PER_S = 760               # from a 50-step test: the new rows and the bigger head cost a little
def same_text_tokens(llama_tokens, fert_llama, fert_ext):
    # TODO: the same documents, counted in extended-vocab tokens
    return llama_tokens * fert_ext / fert_llama


def run_plan(tokens, tokens_per_step, tok_per_s):
    # TODO: optimizer steps for one pass over those tokens
    steps = int(tokens // tokens_per_step)
    # TODO: GPU hours at that throughput
    hours = tokens / tok_per_s / 3600
    return steps, hours


tokens = same_text_tokens(CONTROL_TOKENS, FERT_LLAMA, FERT_EXT)
steps, hours = run_plan(tokens, TOKENS_PER_STEP, TOK_PER_S)
c_steps, c_hours = run_plan(CONTROL_TOKENS, TOKENS_PER_STEP, CONTROL_TOK_PER_S)
print(f"control   {CONTROL_TOKENS / 1e6:5.1f}M Llama tokens     {c_steps:>5} steps  {c_hours:4.1f} GPU hours")
print(f"extended  {tokens / 1e6:5.1f}M extended tokens  {steps:>5} steps  {hours:4.1f} GPU hours")


def lr_at(step, max_lr, warmup, total, min_lr):
    # TODO: warmup, a straight line up to max_lr over the first `warmup` steps
    if step < warmup:
        return max_lr * (step + 1) / warmup
    # TODO: from `total` on, stay at min_lr
    if step >= total:
        return min_lr
    # TODO: in between, half a cosine from max_lr down to min_lr
    progress = (step - warmup) / (total - warmup)
    return max_lr - (max_lr - min_lr) * progress


warmup = 50
print(f"\n{steps} steps, warmup {warmup}, LoRA lr 1e-4 down to 1e-5")
for s in (0, 25, 50, steps // 4, steps // 2, 3 * steps // 4, steps - 1, steps + 100):
    lr = lr_at(s, 1e-4, warmup, steps, 1e-5)
    print(f"step {s:>5}  lr {lr:.2e}  " + "#" * round(lr / 1e-4 * 40))
