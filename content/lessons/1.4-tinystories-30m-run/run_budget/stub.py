
def run_budget(n_params, tokens, peak_tflops, mfu):
    # TODO: training compute, about 6 FLOPs per parameter per token
    flops = ...
    # TODO: how long that takes at the speed you really get: a fraction (mfu) of the GPU's peak
    seconds = ...
    return flops, seconds / 3600

tokens = 15_000 * 32 * 512   # steps × sequences per step × tokens per sequence
flops, hours = run_budget(n, tokens, peak_tflops=71, mfu=0.30)
print(f"{tokens / 1e6:.1f}M tokens  ·  {flops:.2e} FLOPs  ·  {hours:.2f} GPU-hours on an RTX 3090")
