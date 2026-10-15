import torch

X = torch.tensor([
    [0.42, -0.61, -18.5, 0.27, 0.95, 0.33],
    [-0.88, 0.14, -21.0, -0.52, 0.31, 9.6],
    [0.35, 0.72, -19.2, 0.60, -0.44, 0.21],
    [-0.27, -0.95, -17.8, 0.18, 0.66, 8.7],
])
W = torch.tensor([
    [0.21, -0.43, 0.05, 0.37, -0.12, 0.08],
    [-0.30, 0.18, -0.07, 0.25, 0.41, -0.06],
    [0.15, 0.33, 0.04, -0.28, 0.19, 0.11],
])

def quantize_rows(t):
    # 3.1's absmax int8, one scale per row
    scale = t.abs().amax(dim=1, keepdim=True) / 127
    q = torch.round(t / scale).clamp(-127, 127).to(torch.int8)
    return q, scale

def int8_matmul(X, W):
    qx, sx = quantize_rows(X)   # one scale per token
    qw, sw = quantize_rows(W)   # one scale per output channel
    # TODO: multiply the integers, accumulating in int32 as int8 hardware does
    acc = qx.int() @ qw.int().T
    # TODO: scale back: output [i, j] needs token i's scale and channel j's scale
    return acc.float() * torch.outer(sx.flatten(), sw.flatten())

Y = X @ W.T
print(quantize_rows(X)[0])
err = (Y - int8_matmul(X, W)).abs()
print(f"int8   mean err {err.mean().item():.4f}   max err {err.max().item():.4f}")
