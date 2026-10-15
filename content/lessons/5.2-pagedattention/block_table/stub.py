def slot_of(block_table, pos):
    # TODO: find the logical block that holds pos, then look up where it lives
    physical = ...
    # TODO: where pos sits inside that block
    offset = ...
    return physical, offset


def gather_kv(pool, block_table, length):
    # TODO: the request's blocks in logical order, laid end to end, cut to its length
    return ...


for pos in range(10):
    b, o = slot_of(block_table, pos)
    k_pool[b, o] = keys[pos]
back = gather_kv(k_pool, block_table, 10)
print(back.shape, "matches the keys:", torch.equal(back, keys))
