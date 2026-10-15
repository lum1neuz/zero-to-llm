
def gptq(W, H, scale):
    W = W.clone()                     # GPTQ edits a copy as it goes
    Q = torch.zeros_like(W)
    U = upper_cholesky(inverse(H))
    for j in range(W.shape[1]):
        # TODO: round column j onto each row's int4 grid (±7), back in the weights' units
        q = ...
        Q[:, j] = q
        # TODO: how far column j moved, divided by U[j, j]
        err = ...
        # TODO: the correction for columns j onward: err (one per row) times row j of U
        update = ...
        W[:, j:] -= update
    return Q

Q = gptq(W, H, scale)
W_rtn = torch.round(W / scale[:, None]).clamp(-7, 7) * scale[:, None]
for name, Wq in (("RTN ", W_rtn), ("GPTQ", Q)):
    out_err = ((X @ W.T - X @ Wq.T) ** 2).mean()
    w_err = (W - Wq).abs().mean()
    print(f"{name}  output err {out_err.item():.5f}   weight err {w_err.item():.4f}")
