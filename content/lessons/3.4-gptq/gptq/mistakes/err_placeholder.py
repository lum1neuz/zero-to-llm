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

scale = W.abs().amax(dim=1) / 7   # one int4 scale per row

def inverse(A):
    # Gauss-Jordan elimination. On your GPU: torch.linalg.inv(A)
    n = A.shape[0]
    M = torch.cat([A.clone(), torch.eye(n)], dim=1)
    for i in range(n):
        M[i] = M[i] / M[i, i]
        for r in range(n):
            if r != i:
                M[r] = M[r] - M[r, i] * M[i]
    return M[:, n:]

def upper_cholesky(A):
    # U with A = U.T @ U. On your GPU: torch.linalg.cholesky(A, upper=True)
    n = A.shape[0]
    U = torch.zeros_like(A)
    for i in range(n):
        U[i, i] = (A[i, i] - (U[:i, i] ** 2).sum()).sqrt()
        for k in range(i + 1, n):
            U[i, k] = (A[i, k] - (U[:i, i] * U[:i, k]).sum()) / U[i, i]
    return U

def gptq(W, H, scale):
    W = W.clone()                     # GPTQ edits a copy as it goes
    Q = torch.zeros_like(W)
    U = upper_cholesky(inverse(H))
    for j in range(W.shape[1]):
        # TODO: round column j onto each row's int4 grid (±7), back in the weights' units
        q = torch.round(W[:, j] / scale).clamp(-7, 7) * scale
        Q[:, j] = q
        # TODO: how far column j moved, divided by U[j, j]
        err = ...
        # TODO: the correction for columns j onward: err (one per row) times row j of U
        update = err[:, None] * U[j, j:]
        W[:, j:] -= update
    return Q

Q = gptq(W, H, scale)
W_rtn = torch.round(W / scale[:, None]).clamp(-7, 7) * scale[:, None]
for name, Wq in (("RTN ", W_rtn), ("GPTQ", Q)):
    out_err = ((X @ W.T - X @ Wq.T) ** 2).mean()
    w_err = (W - Wq).abs().mean()
    print(f"{name}  output err {out_err.item():.5f}   weight err {w_err.item():.4f}")
