

def attention(q, k, v):
    # q, k, v: [B, H, T, D]. Causal attention written out, the way 1.1 did it.
    T, D = q.shape[-2], q.shape[-1]
    # TODO: every query against every key, scaled by sqrt(D): shape [B, H, T, T]
    scores = ...
    # TODO: hide the future, so query t can't see key s > t
    masked = ...
    # TODO: softmax over the keys, then mix the values
    return ...


g = torch.Generator().manual_seed(0)
q, k, v = (torch.randn(2, 4, 16, 8, generator=g) for _ in range(3))
mine = attention(q, k, v)
fused = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("max diff vs SDPA:", (mine - fused).abs().max().item())
B, H, T, D = q.shape
print(f"your scores tensor: {B * H * T * T * 4:,} bytes. SDPA's flash kernel on a GPU never writes it to memory.")
