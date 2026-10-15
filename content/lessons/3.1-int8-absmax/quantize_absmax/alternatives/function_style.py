import torch

w = torch.tensor([0.42, -1.30, 0.07, 0.88, -0.25, 2.10, -0.61, 0.33])

def quantize_absmax(w):
    # TODO: pick scale so the largest |w| lands on 127
    scale = w.abs().max() / 127
    # TODO: divide, round, keep inside [-127, 127], store as int8
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale
