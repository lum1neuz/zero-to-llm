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

import torch.nn.functional as F

def paged_attention(q, k_pool, v_pool, block_table, length):
    # TODO: this request's cached keys and values, gathered from their pools
    k = gather_kv(k_pool, block_table, len(block_table) * BLOCK)
    v = gather_kv(v_pool, block_table, len(block_table) * BLOCK)
    # TODO: one weight per cached token: softmax over q·k / sqrt(d)
    weights = torch.softmax(k @ q / k.shape[-1] ** 0.5, dim=-1)
    return weights @ v


v_pool = torch.zeros(6, BLOCK, 2)
values = torch.randn(10, 2, generator=torch.Generator().manual_seed(1))
for pos in range(10):
    b, o = slot_of(block_table, pos)
    v_pool[b, o] = values[pos]
q = torch.randn(2, generator=torch.Generator().manual_seed(2))

dense = F.scaled_dot_product_attention(q.view(1, 2), keys, values).view(2)
paged = paged_attention(q, k_pool, v_pool, block_table, 10)
print(f"paged attention vs dense: max abs diff {(paged - dense).abs().max().item():.1e}")
