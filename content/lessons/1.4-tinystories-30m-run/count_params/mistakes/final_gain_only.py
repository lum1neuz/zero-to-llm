config = dict(vocab_size=21504, block_size=512, n_layer=6, n_head=8, n_embd=512)

def count_params(cfg):
    V, T, L, d = cfg["vocab_size"], cfg["block_size"], cfg["n_layer"], cfg["n_embd"]
    # TODO: token embeddings (wte) and position embeddings (wpe)
    embeddings = V * d + T * d
    # TODO: every block, using 1.2's per-block count
    blocks = L * (12 * d * d + 13 * d)
    # TODO: the final LayerNorm. lm_head shares wte's weights, so it adds nothing
    final = d
    return embeddings + blocks + final

n = count_params(config)
print(f"{n:,} params  ·  fp16 weights {n * 2 / 1e6:.1f} MB")
