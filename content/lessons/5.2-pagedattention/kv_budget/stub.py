def kv_bytes_per_token(cfg, bytes_per_value):
    # TODO: a K and a V vector for every KV head, in every layer
    return ...


def waste_pct(lengths, max_len, block_size):
    # TODO: contiguous: every request reserves max_len slots up front
    contiguous_reserved = ...
    # TODO: paged: every request reserves whole blocks, just enough to hold its tokens
    paged_reserved = ...
    used = sum(lengths)
    return 100 * (1 - used / contiguous_reserved), 100 * (1 - used / paged_reserved)


lengths = [612, 37, 1450, 220, 903, 75, 3010, 488]   # tokens each request holds right now
per_token = kv_bytes_per_token(cfg, 2)
print(f"KV per token {per_token / 1024:.0f} KiB   one 4096-token request {per_token * 4096 / 2**30:.2f} GiB")
contiguous, paged = waste_pct(lengths, 4096, 16)
print(f"reserved but empty: contiguous {contiguous:.1f}%   paged {paged:.1f}%")
