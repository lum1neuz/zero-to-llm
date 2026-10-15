import torch
import torch.nn as nn
import torch.nn.functional as F

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
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # TODO: put the heads back side by side: [B, n_head, T, hs] -> [B, T, C]
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.c_proj(y)

torch.manual_seed(0)
attn = CausalSelfAttention(n_embd=32, n_head=4)
x = torch.randn(2, 8, 32)
print("attn out", tuple(attn(x).shape))

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        # TODO: one LayerNorm in front of attention, another in front of the MLP
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        # TODO: each sublayer reads a normalised copy of x and adds its result back onto x
        x = x + self.ln_1(self.attn(x))
        x = x + self.mlp(self.ln_2(x))
        return x

block = Block(n_embd=32, n_head=4)
print("block out", tuple(block(x).shape))
print("params", sum(p.numel() for p in block.parameters()))
