
def dequantize(q, scale):
    # TODO: undo the quantization
    return ...

q, scale = quantize_absmax(w)
w_hat = dequantize(q, scale)
print("q      ", q.tolist())
print("max err", (w - w_hat).abs().max().item())
print("bound  ", scale.item() / 2)
