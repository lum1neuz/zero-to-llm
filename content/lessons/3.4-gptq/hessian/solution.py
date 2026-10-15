import torch

g = torch.Generator().manual_seed(6)
mix = torch.randn(3, 8, generator=g)
X = torch.randn(64, 3, generator=g) @ mix + 0.1 * torch.randn(64, 8, generator=g)
X[:, 7] = 0.0
W = 0.3 * torch.randn(4, 8, generator=g)

def hessian(X, percdamp=0.01):
    idx = torch.arange(X.shape[1])   # 0, 1, …, features − 1
    # TODO: H = 2 XᵀX / n, one row and one column per feature
    H = 2 * X.T @ X / X.shape[0]
    # TODO: percdamp times the mean of H's diagonal
    damp = percdamp * H[idx, idx].mean()
    # TODO: add damp to the diagonal only
    return H + damp * torch.eye(len(idx))

H = hessian(X)
print(H)
