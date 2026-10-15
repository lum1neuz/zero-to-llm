
def dequantize_groups(q, scale):
    # TODO: undo the quantization, then flatten the groups back into one row
    return ...

for g in (16, 4):
    q, scale = quantize_groups(w, g)
    w_hat = dequantize_groups(q, scale)
    err = (w - w_hat).abs()
    print(f"g={g:<2}  scales {scale.shape[0]:>2}  mean err {err.mean().item():.4f}  max err {err.max().item():.4f}")
