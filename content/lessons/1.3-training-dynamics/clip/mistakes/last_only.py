import math

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(step):
    # TODO: warmup: climb in a straight line, so the last warmup step reaches max_lr
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # TODO: after max_steps, hold at the floor
    if step > max_steps:
        return min_lr
    # TODO: in between, how far through the decay are we (0 → 1), and half a cosine from 1 down to 0
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)

for step in (0, 4, 9, 10, 30, 50, 60):
    print(f"step {step:>2}  lr {get_lr(step):.2e}")

import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))
loss = (model(torch.randn(32, 4)) * 20).pow(2).mean()   # a deliberately bad batch
loss.backward()

def clip_grad_norm(params, max_norm):
    # TODO: one norm for ALL the gradients, as if they were laid end to end in one long vector
    for p in params:
        total_norm = p.grad.norm()
    # TODO: the factor that brings total_norm down to max_norm, but never scales up
    scale = min(1.0, max_norm / (float(total_norm) + 1e-6))
    for p in params:
        # TODO: scale this gradient in place
        p.grad.mul_(scale)
    return total_norm

params = list(model.parameters())
before = clip_grad_norm(params, 1.0)
after = sum((p.grad ** 2).sum() for p in params) ** 0.5
print(f"grad norm before {float(before):.2f}  after {float(after):.4f}")
