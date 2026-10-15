
def hessian(X, percdamp=0.01):
    idx = torch.arange(X.shape[1])   # 0, 1, …, features − 1
    # TODO: H = 2 XᵀX / n, one row and one column per feature
    H = ...
    # TODO: percdamp times the mean of H's diagonal
    damp = ...
    # TODO: add damp to the diagonal only
    return ...

H = hessian(X)
print(H)
