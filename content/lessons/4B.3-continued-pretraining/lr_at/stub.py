

def lr_at(step, max_lr, warmup, total, min_lr):
    # TODO: warmup, a straight line up to max_lr over the first `warmup` steps
    if step < warmup:
        return ...
    # TODO: from `total` on, stay at min_lr
    if step >= total:
        return ...
    # TODO: in between, half a cosine from max_lr down to min_lr
    progress = ...
    return ...


warmup = 50
print(f"\n{steps} steps, warmup {warmup}, LoRA lr 1e-4 down to 1e-5")
for s in (0, 25, 50, steps // 4, steps // 2, 3 * steps // 4, steps - 1, steps + 100):
    lr = lr_at(s, 1e-4, warmup, steps, 1e-5)
    print(f"step {s:>5}  lr {lr:.2e}  " + "#" * round(lr / 1e-4 * 40))
