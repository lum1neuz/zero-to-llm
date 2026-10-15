# Llama 3.1 8B: 32 query heads share 8 KV heads (grouped-query attention)
cfg = {"layers": 32, "heads": 32, "kv_heads": 8, "head_dim": 128}

def kv_bytes_per_token(cfg, bytes_per_value):
    # TODO: a K and a V vector for every KV head, in every layer
    return 2 * cfg["layers"] * cfg["kv_heads"] * cfg["head_dim"] * bytes_per_value


def waste_pct(lengths, max_len, block_size):
    # TODO: contiguous: every request reserves max_len slots up front
    contiguous_reserved = len(lengths) * max_len
    # TODO: paged: every request reserves whole blocks, just enough to hold its tokens
    paged_reserved = sum((n + block_size - 1) // block_size * block_size for n in lengths)
    used = sum(lengths)
    return 100 * (1 - used / contiguous_reserved), 100 * (1 - used / paged_reserved)


lengths = [612, 37, 1450, 220, 903, 75, 3010, 488]   # tokens each request holds right now
