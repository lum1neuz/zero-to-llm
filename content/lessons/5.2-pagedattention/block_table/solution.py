import torch

BLOCK = 4   # tokens per block

# This request's logical block 0 lives in physical block 3, block 1 in block 0, block 2 in block 5
block_table = [3, 0, 5]

# Six physical blocks, shared by every request, and this request's 10 keys (d = 2)
k_pool = torch.zeros(6, BLOCK, 2)
keys = torch.randn(10, 2, generator=torch.Generator().manual_seed(0))

def slot_of(block_table, pos):
    # TODO: find the logical block that holds pos, then look up where it lives
    physical = block_table[pos // BLOCK]
    # TODO: where pos sits inside that block
    offset = pos % BLOCK
    return physical, offset


def gather_kv(pool, block_table, length):
    # TODO: the request's blocks in logical order, laid end to end, cut to its length
    return pool[block_table].reshape(-1, pool.shape[-1])[:length]


for pos in range(10):
    b, o = slot_of(block_table, pos)
    k_pool[b, o] = keys[pos]
back = gather_kv(k_pool, block_table, 10)
print(back.shape, "matches the keys:", torch.equal(back, keys))
