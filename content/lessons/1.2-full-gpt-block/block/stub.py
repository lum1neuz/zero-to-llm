
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
        self.ln_1 = ...
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = ...
        self.mlp = MLP(n_embd)

    def forward(self, x):
        # TODO: each sublayer reads a normalised copy of x and adds its result back onto x
        x = ...
        x = ...
        return x

block = Block(n_embd=32, n_head=4)
print("block out", tuple(block(x).shape))
print("params", sum(p.numel() for p in block.parameters()))
