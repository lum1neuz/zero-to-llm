import torch

w = 0.05 * torch.randn(256, generator=torch.Generator().manual_seed(0))

def int4_blocks(w, block):
    wb = w.reshape(-1, block)
    scale = wb.abs().amax(dim=1, keepdim=True) / 7
    return (torch.round(wb / scale).clamp(-7, 7) * scale).reshape(-1)

def quantize_subblocks(w):
    wb = w.reshape(-1, 32)                # 8 sub-blocks of 32
    lo = wb.amin(dim=1, keepdim=True)
    hi = wb.amax(dim=1, keepdim=True)
    # TODO: the step that spreads lo…hi over the 16 codes 0…15
    scale = (hi - lo) / 15
    # TODO: each weight's code, counted up from lo, stored in one unsigned byte
    q = torch.round((wb - lo) / scale).clamp(0, 15).to(torch.uint8)
    return q, scale, lo

q, scale, lo = quantize_subblocks(w)
w_hat = (q * scale + lo).reshape(-1)    # the way back: code times scale, plus lo
print(f"int4 absmax, 15 levels around 0   mean err {(w - int4_blocks(w, 32)).abs().mean().item():.5f}")
print(f"scale + min, 16 levels lo…hi      mean err {(w - w_hat).abs().mean().item():.5f}")

def six_bit(v):
    # TODO: one super-scale for all the values, so the largest lands on 63 (6 bits)
    d = ...
    # TODO: each value as a whole number of d's, 0…63
    q = ...
    return q.to(torch.uint8), d

sc, d = six_bit(scale)        # the 8 sub-block scales
mq, dmin = six_bit(-lo)       # the 8 mins, stored as positive numbers
scale_k, lo_k = d * sc, -dmin * mq
q_k = torch.round((w.reshape(-1, 32) - lo_k) / scale_k).clamp(0, 15)   # llama.cpp re-fits the codes to the rounded scales
w_k = (q_k * scale_k + lo_k).reshape(-1)
bits = (256 * 4 + 8 * 6 + 8 * 6 + 2 * 16) / 256
print(f"fp16 scale + min per 32   {4 + 2 * 16 / 32:.3f} bits/weight   mean err {(w - w_hat).abs().mean().item():.5f}")
print(f"Q4_K super-block          {bits:.3f} bits/weight   mean err {(w - w_k).abs().mean().item():.5f}")
