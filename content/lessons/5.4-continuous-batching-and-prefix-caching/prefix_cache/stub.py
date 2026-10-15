def block_hashes(tokens, block_size):
    hashes = []
    parent = None
    # TODO: one hash per FULL block. Each block's hash covers its parent's hash and its own tokens,
    # so it stands for everything from the start of the prompt to the end of that block.
    ...
    return hashes


def cached_prefix_len(tokens, cache, block_size):
    # TODO: walk the blocks from the start, adding block_size tokens while each block's hash is in cache.
    # Stop at the first miss.
    ...


cache = set(block_hashes(first, BLOCK))   # after serving the first request, its full blocks stay cached
print(f"first:  {len(first)} tokens, {len(block_hashes(first, BLOCK))} full blocks cached")
hit = cached_prefix_len(second, cache, BLOCK)
print(f"second: {hit} of {len(second)} tokens reuse cached KV, {len(second) - hit} left to prefill")
