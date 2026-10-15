import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
base = nn.Linear(6, 10)
x = torch.randn(3, 6)

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        # TODO: freeze every parameter of the base layer
        self.base.requires_grad_(False)
        # TODO: A is [r, in] with small random values, B is [out, r] of zeros. Both train.
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        # TODO: how much the update counts, from alpha and r
        self.scale = alpha / r

    def forward(self, x):
        # TODO: the frozen layer's output plus the scaled low-rank update
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


lora = LoRALinear(base, r=4, alpha=8)
print("same as base at step 0:", torch.equal(lora(x), base(x)))
print("trainable:", [name for name, p in lora.named_parameters() if p.requires_grad])
