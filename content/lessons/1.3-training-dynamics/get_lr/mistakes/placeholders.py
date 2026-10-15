import math

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(step):
    # TODO: warmup: climb in a straight line, so the last warmup step reaches max_lr
    if step < warmup_steps:
        return ...
    # TODO: after max_steps, hold at the floor
    if step > max_steps:
        return ...
    # TODO: in between, how far through the decay are we (0 → 1), and half a cosine from 1 down to 0
    progress = ...
    coeff = ...
    return min_lr + coeff * (max_lr - min_lr)

for step in (0, 4, 9, 10, 30, 50, 60):
    print(f"step {step:>2}  lr {get_lr(step):.2e}")
