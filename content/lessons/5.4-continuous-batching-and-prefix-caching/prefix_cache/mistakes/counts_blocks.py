BLOCK = 4
system = [101, 7, 7, 42, 9, 13, 250, 3, 3, 88]   # a 10-token system prompt, shared by every request
first = system + [5, 61, 17, 2, 40, 40]          # the first user's request
second = system + [5, 61, 99, 8, 12]             # a second user, same system prompt

def block_hashes(tokens, block_size):
    hashes = []
    parent = None
    # TODO: one hash per FULL block. Each block's hash covers its parent's hash and its own tokens,
    # so it stands for everything from the start of the prompt to the end of that block.
    for i in range(len(tokens) // block_size):
        block = tuple(tokens[i * block_size:(i + 1) * block_size])
        parent = hash((parent, block))
        hashes.append(parent)
    return hashes


def cached_prefix_len(tokens, cache, block_size):
    # TODO: walk the blocks from the start, adding block_size tokens while each block's hash is in cache.
    # Stop at the first miss.
    n = 0
    for h in block_hashes(tokens, block_size):
        if h not in cache:
            break
        n += 1
    return n


cache = set(block_hashes(first, BLOCK))   # after serving the first request, its full blocks stay cached
print(f"first:  {len(first)} tokens, {len(block_hashes(first, BLOCK))} full blocks cached")
hit = cached_prefix_len(second, cache, BLOCK)
print(f"second: {hit} of {len(second)} tokens reuse cached KV, {len(second) - hit} left to prefill")
