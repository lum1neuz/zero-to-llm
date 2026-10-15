def step_ms(n_params, bytes_per_param, batch):
    # TODO: ms to read every weight from GPU memory once
    mem_ms = ...
    # TODO: ms of math: 2 FLOPs per parameter, for each sequence in the batch
    compute_ms = ...
    # TODO: the GPU reads and computes at the same time, so the step lasts as long as the slower one
    return ...


def tokens_per_s(n_params, bytes_per_param, batch):
    # TODO: each step makes one new token for every sequence in the batch
    return ...


for batch in (1, 8, 64, 256):
    print(f"batch {batch:>3}   step {step_ms(8e9, 2, batch):6.2f} ms   {tokens_per_s(8e9, 2, batch):6.0f} tok/s")
