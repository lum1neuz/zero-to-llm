
def nf4_quantize(w, block):
    wb = w.reshape(-1, block)
    absmax = wb.abs().amax(dim=1, keepdim=True)   # one absmax per block
    # TODO: scale every block into [-1, 1]
    wn = ...
    # TODO: for each weight, the index (0…15) of the nearest NF4 level
    idx = ...
    return idx.to(torch.uint8), absmax

def nf4_dequantize(idx, absmax):
    # TODO: look the levels up, undo the block scaling, flatten back to one row
    return ...

for name, w_hat in (("int4 absmax", int4_blocks(w, 64)), ("NF4", nf4_dequantize(*nf4_quantize(w, 64)))):
    print(f"{name:<12} mean err {(w - w_hat).abs().mean().item():.5f}")
