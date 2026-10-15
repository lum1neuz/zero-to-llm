
def smooth(X, W, alpha=0.5):
    # TODO: one factor per feature: max|X_j|^alpha / max|W_j|^(1 - alpha)
    s = ...
    # TODO: activations divided by s, weights multiplied by s, so X @ W.T stays the same
    return ...

X_s, W_s = smooth(X, W)
print(quantize_rows(X_s)[0])
err = (Y - int8_matmul(X_s, W_s)).abs()
print(f"smooth mean err {err.mean().item():.4f}   max err {err.max().item():.4f}")
