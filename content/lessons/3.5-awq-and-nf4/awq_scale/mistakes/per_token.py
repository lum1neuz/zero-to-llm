import torch

g = torch.Generator().manual_seed(1)
X = torch.randn(32, 8, generator=g) * torch.tensor([1.0, 0.8, 6.0, 1.2, 0.9, 1.0, 3.0, 0.7])
W = 0.3 * torch.randn(4, 8, generator=g)

def quantize_rows_int4(W):
    scale = W.abs().amax(dim=1, keepdim=True) / 7
    return torch.round(W / scale).clamp(-7, 7) * scale

def awq(W, X, alpha):
    # TODO: one factor per input channel: its mean |x| over the tokens, to the power alpha
    s = X.abs().mean(dim=1) ** alpha
    # TODO: stretch W's columns by s, quantize, then shrink them back
    return quantize_rows_int4(W * s) / s

for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
    W_hat = awq(W, X, alpha)
    err = ((X @ W.T - X @ W_hat.T) ** 2).mean()
    print(f"alpha {alpha:.2f}  output err {err.item():.5f}")
