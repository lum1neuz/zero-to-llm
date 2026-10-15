config = dict(vocab_size=21504, block_size=512, n_layer=6, n_head=8, n_embd=512)

def count_params(cfg):
    V, T, L, d = cfg["vocab_size"], cfg["block_size"], cfg["n_layer"], cfg["n_embd"]
    # TODO: token embeddings (wte) and position embeddings (wpe)
    wte = V * d
    wpe = T * d
    embeddings = wte + wpe
    # TODO: every block, using 1.2's per-block count
    per_block = 4 * d * d + 4 * d + 8 * d * d + 5 * d + 4 * d
    blocks = L * per_block
    # TODO: the final LayerNorm. lm_head shares wte's weights, so it adds nothing
    final = 2 * d
    return embeddings + blocks + final

n = count_params(config)
print(f"{n:,} params  ·  fp16 weights {n * 2 / 1e6:.1f} MB")
