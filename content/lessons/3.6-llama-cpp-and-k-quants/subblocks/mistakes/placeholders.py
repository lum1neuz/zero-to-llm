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
    scale = ...
    # TODO: each weight's code, counted up from lo, stored in one unsigned byte
    q = ...
    return q, scale, lo

q, scale, lo = quantize_subblocks(w)
w_hat = (q * scale + lo).reshape(-1)    # the way back: code times scale, plus lo
print(f"int4 absmax, 15 levels around 0   mean err {(w - int4_blocks(w, 32)).abs().mean().item():.5f}")
print(f"scale + min, 16 levels lo…hi      mean err {(w - w_hat).abs().mean().item():.5f}")
