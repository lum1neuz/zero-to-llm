import torch

w = 0.05 * torch.randn(512, generator=torch.Generator().manual_seed(0))

NF4 = torch.tensor([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
                    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0])

def int4_blocks(w, block):
    wb = w.reshape(-1, block)
    scale = wb.abs().amax(dim=1, keepdim=True) / 7
    return (torch.round(wb / scale).clamp(-7, 7) * scale).reshape(-1)

def nf4_quantize(w, block):
    wb = w.reshape(-1, block)
    absmax = wb.abs().amax(dim=1, keepdim=True)   # one absmax per block
    # TODO: scale every block into [-1, 1]
    wn = wb
    # TODO: for each weight, the index (0…15) of the nearest NF4 level
    idx = (wn[..., None] - NF4).abs().argmin(dim=-1)
    return idx.to(torch.uint8), absmax

def nf4_dequantize(idx, absmax):
    # TODO: look the levels up, undo the block scaling, flatten back to one row
    return (NF4[idx.long()] * absmax).reshape(-1)

for name, w_hat in (("int4 absmax", int4_blocks(w, 64)), ("NF4", nf4_dequantize(*nf4_quantize(w, 64)))):
    print(f"{name:<12} mean err {(w - w_hat).abs().mean().item():.5f}")
