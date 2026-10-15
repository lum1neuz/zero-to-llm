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
    return acc.float() * sx * sw.T

Y = X @ W.T
print(quantize_rows(X)[0])
err = (Y - int8_matmul(X, W)).abs()
print(f"int8   mean err {err.mean().item():.4f}   max err {err.max().item():.4f}")

def mixed_matmul(X, W, threshold=6.0):
    # TODO: one flag per feature column: True where any token's |x| passes the threshold
    big = (X.abs() > threshold).any(dim=0)
    # TODO: outlier columns in full precision, every other column through int8_matmul
    return X[:, big] @ W[:, big].T + int8_matmul(X[:, ~big], W[:, ~big])

err = (Y - mixed_matmul(X, W)).abs()
print(f"mixed  mean err {err.mean().item():.4f}   max err {err.max().item():.4f}")

def smooth(X, W, alpha=0.5):
    # TODO: one factor per feature: max|X_j|^alpha / max|W_j|^(1 - alpha)
    s = X.abs().amax(dim=0).pow(alpha) / W.abs().amax(dim=0).pow(1 - alpha)
    # TODO: activations divided by s, weights multiplied by s, so X @ W.T stays the same
    return X / s, W * s

X_s, W_s = smooth(X, W)
print(quantize_rows(X_s)[0])
err = (Y - int8_matmul(X_s, W_s)).abs()
print(f"smooth mean err {err.mean().item():.4f}   max err {err.max().item():.4f}")
