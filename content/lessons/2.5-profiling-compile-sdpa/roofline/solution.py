import torch
import torch.nn.functional as F

rtx3090 = dict(peak_tflops=71, bandwidth_gbs=936)  # dense fp16 tensor-core peak (fp32 accumulate), memory bandwidth

def decode_step_ms(params, batch, bytes_per_param, peak_tflops, bandwidth_gbs):
    # TODO: a forward pass costs about 2 FLOPs per parameter, for every token in the batch
    flops = 2 * params * batch
    # TODO: every weight is read from memory once per step, however big the batch
    bytes_moved = params * bytes_per_param
    compute_ms = flops / (peak_tflops * 1e12) * 1e3
    memory_ms = bytes_moved / (bandwidth_gbs * 1e9) * 1e3
    # TODO: the step is done only when both are done. Which one is it waiting on?
    ms = max(compute_ms, memory_ms)
    bound = "compute" if compute_ms > memory_ms else "memory"
    return ms, bound


for batch in (1, 16, 64, 256):
    ms, bound = decode_step_ms(8e9, batch, 2, **rtx3090)
    print(f"Llama 3.1 8B fp16, batch {batch:>3}: {ms:5.1f} ms/step  {batch / ms * 1e3:6.0f} tok/s  {bound}-bound")
