
import torch.nn.functional as F

def paged_attention(q, k_pool, v_pool, block_table, length):
    # TODO: this request's cached keys and values, gathered from their pools
    k = ...
    v = ...
    # TODO: one weight per cached token: softmax over q·k / sqrt(d)
    weights = ...
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
