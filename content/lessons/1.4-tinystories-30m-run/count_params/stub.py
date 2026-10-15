def count_params(cfg):
    V, T, L, d = cfg["vocab_size"], cfg["block_size"], cfg["n_layer"], cfg["n_embd"]
    # TODO: token embeddings (wte) and position embeddings (wpe)
    embeddings = ...
    # TODO: every block, using 1.2's per-block count
    blocks = ...
    # TODO: the final LayerNorm. lm_head shares wte's weights, so it adds nothing
    final = ...
    return embeddings + blocks + final

n = count_params(config)
print(f"{n:,} params  ·  fp16 weights {n * 2 / 1e6:.1f} MB")
