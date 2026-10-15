# RTX 3090, rounded spec-sheet numbers
BANDWIDTH = 936e9    # bytes per second from GPU memory
PEAK_FLOPS = 71e12   # fp16 tensor-core FLOPs per second

def step_ms(n_params, bytes_per_param, batch):
    # TODO: ms to read every weight from GPU memory once
    mem_ms = n_params * bytes_per_param / BANDWIDTH * 1000
    # TODO: ms of math: 2 FLOPs per parameter, for each sequence in the batch
    compute_ms = 2 * n_params * batch / PEAK_FLOPS * 1000
    # TODO: the GPU reads and computes at the same time, so the step lasts as long as the slower one
    return min(mem_ms, compute_ms)


def tokens_per_s(n_params, bytes_per_param, batch):
    # TODO: each step makes one new token for every sequence in the batch
    return batch * 1000 / step_ms(n_params, bytes_per_param, batch)


for batch in (1, 8, 64, 256):
    print(f"batch {batch:>3}   step {step_ms(8e9, 2, batch):6.2f} ms   {tokens_per_s(8e9, 2, batch):6.0f} tok/s")
