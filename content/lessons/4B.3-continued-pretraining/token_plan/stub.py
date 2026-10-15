def same_text_tokens(llama_tokens, fert_llama, fert_ext):
    # TODO: the same documents, counted in extended-vocab tokens
    return ...


def run_plan(tokens, tokens_per_step, tok_per_s):
    # TODO: optimizer steps for one pass over those tokens
    steps = ...
    # TODO: GPU hours at that throughput
    hours = ...
    return steps, hours


tokens = same_text_tokens(CONTROL_TOKENS, FERT_LLAMA, FERT_EXT)
steps, hours = run_plan(tokens, TOKENS_PER_STEP, TOK_PER_S)
c_steps, c_hours = run_plan(CONTROL_TOKENS, TOKENS_PER_STEP, CONTROL_TOK_PER_S)
print(f"control   {CONTROL_TOKENS / 1e6:5.1f}M Llama tokens     {c_steps:>5} steps  {c_hours:4.1f} GPU hours")
print(f"extended  {tokens / 1e6:5.1f}M extended tokens  {steps:>5} steps  {hours:4.1f} GPU hours")
