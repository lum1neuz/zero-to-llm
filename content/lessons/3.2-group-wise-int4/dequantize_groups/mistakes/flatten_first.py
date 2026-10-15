import torch

w = torch.tensor([0.12, -0.31, 0.05, 0.22, -0.18, 0.09, 2.40, -0.27, 0.15, -0.08, 0.30, -0.11, -0.25, 0.19, -0.04, 0.28])

def quantize_groups(w, g):
    # TODO: lay the row out as groups of g, one group per row
    wg = w.reshape(-1, g)
    # TODO: one absmax scale per group, so its largest |w| lands on 7
    scale = wg.abs().amax(dim=1, keepdim=True) / 7
    # TODO: divide, round, keep inside [-7, 7], store as int8
    q = torch.round(wg / scale).clamp(-7, 7).to(torch.int8)
    return q, scale

def dequantize_groups(q, scale):
    # TODO: undo the quantization, then flatten the groups back into one row
    return q.float().reshape(-1) * scale

for g in (16, 4):
    q, scale = quantize_groups(w, g)
    w_hat = dequantize_groups(q, scale)
    err = (w - w_hat).abs()
    print(f"g={g:<2}  scales {scale.shape[0]:>2}  mean err {err.mean().item():.4f}  max err {err.max().item():.4f}")
