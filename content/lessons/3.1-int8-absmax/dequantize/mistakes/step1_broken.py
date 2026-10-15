import torch

w = torch.tensor([0.42, -1.30, 0.07, 0.88, -0.25, 2.10, -0.61, 0.33])

def quantize_absmax(w):
    # TODO: pick scale so the largest |w| lands on 127
    scale = w.max() / 127
    # TODO: divide, round, keep inside [-127, 127], store as int8
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q, scale

def dequantize(q, scale):
    # TODO: undo the quantization
    return q.float() * scale

q, scale = quantize_absmax(w)
w_hat = dequantize(q, scale)
print("q      ", q.tolist())
print("max err", (w - w_hat).abs().max().item())
print("bound  ", scale.item() / 2)
