
def mixed_matmul(X, W, threshold=6.0):
    # TODO: one flag per feature column: True where any token's |x| passes the threshold
    big = ...
    # TODO: outlier columns in full precision, every other column through int8_matmul
    return ...

err = (Y - mixed_matmul(X, W)).abs()
print(f"mixed  mean err {err.mean().item():.4f}   max err {err.max().item():.4f}")
