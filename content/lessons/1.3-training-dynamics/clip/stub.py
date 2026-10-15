
import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))
loss = (model(torch.randn(32, 4)) * 20).pow(2).mean()   # a deliberately bad batch
loss.backward()

def clip_grad_norm(params, max_norm):
    # TODO: one norm for ALL the gradients, as if they were laid end to end in one long vector
    total_norm = ...
    # TODO: the factor that brings total_norm down to max_norm, but never scales up
    scale = ...
    for p in params:
        # TODO: scale this gradient in place
        ...
    return total_norm

params = list(model.parameters())
before = clip_grad_norm(params, 1.0)
after = sum((p.grad ** 2).sum() for p in params) ** 0.5
print(f"grad norm before {float(before):.2f}  after {float(after):.4f}")
