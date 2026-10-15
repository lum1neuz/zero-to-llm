
def awq(W, X, alpha):
    # TODO: one factor per input channel: its mean |x| over the tokens, to the power alpha
    s = ...
    # TODO: stretch W's columns by s, quantize, then shrink them back
    return ...

for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
    W_hat = awq(W, X, alpha)
    err = ((X @ W.T - X @ W_hat.T) ** 2).mean()
    print(f"alpha {alpha:.2f}  output err {err.item():.5f}")
