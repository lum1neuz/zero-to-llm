class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # q, k and v for every head in one matmul
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape
        hs = C // self.n_head
        q, k, v = self.c_attn(x).split(C, dim=2)
        # TODO: give each head its own axis: [B, T, C] -> [B, n_head, T, hs]
        q = ...
        k = ...
        v = ...
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # TODO: put the heads back side by side: [B, n_head, T, hs] -> [B, T, C]
        y = ...
        return self.c_proj(y)

torch.manual_seed(0)
attn = CausalSelfAttention(n_embd=32, n_head=4)
x = torch.randn(2, 8, 32)
print("attn out", tuple(attn(x).shape))
