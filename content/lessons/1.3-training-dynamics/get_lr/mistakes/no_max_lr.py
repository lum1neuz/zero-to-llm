import math

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(step):
    # TODO: warmup: climb in a straight line, so the last warmup step reaches max_lr
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    # TODO: after max_steps, hold at the floor
    if step > max_steps:
        return min_lr
    # TODO: in between, how far through the decay are we (0 → 1), and half a cosine from 1 down to 0
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)

for step in (0, 4, 9, 10, 30, 50, 60):
    print(f"step {step:>2}  lr {get_lr(step):.2e}")
