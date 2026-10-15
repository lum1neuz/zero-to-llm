
def int8_matmul(X, W):
    qx, sx = quantize_rows(X)   # one scale per token
    qw, sw = quantize_rows(W)   # one scale per output channel
    # TODO: multiply the integers, accumulating in int32 as int8 hardware does
    acc = ...
    # TODO: scale back: output [i, j] needs token i's scale and channel j's scale
    return ...

Y = X @ W.T
print(quantize_rows(X)[0])
err = (Y - int8_matmul(X, W)).abs()
print(f"int8   mean err {err.mean().item():.4f}   max err {err.max().item():.4f}")
