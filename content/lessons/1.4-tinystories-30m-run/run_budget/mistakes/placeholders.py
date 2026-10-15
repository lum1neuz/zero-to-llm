config = dict(vocab_size=21504, block_size=512, n_layer=6, n_head=8, n_embd=512)

def count_params(cfg):
    V, T, L, d = cfg["vocab_size"], cfg["block_size"], cfg["n_layer"], cfg["n_embd"]
    # TODO: token embeddings (wte) and position embeddings (wpe)
    embeddings = V * d + T * d
    # TODO: every block, using 1.2's per-block count
    blocks = L * (12 * d * d + 13 * d)
    # TODO: the final LayerNorm. lm_head shares wte's weights, so it adds nothing
    final = 2 * d
    return embeddings + blocks + final

n = count_params(config)
print(f"{n:,} params  ·  fp16 weights {n * 2 / 1e6:.1f} MB")

def run_budget(n_params, tokens, peak_tflops, mfu):
    # TODO: training compute, about 6 FLOPs per parameter per token
    flops = ...
    # TODO: how long that takes at the speed you really get: a fraction (mfu) of the GPU's peak
    seconds = ...
    return flops, seconds / 3600

tokens = 15_000 * 32 * 512   # steps × sequences per step × tokens per sequence
flops, hours = run_budget(n, tokens, peak_tflops=71, mfu=0.30)
print(f"{tokens / 1e6:.1f}M tokens  ·  {flops:.2e} FLOPs  ·  {hours:.2f} GPU-hours on an RTX 3090")
