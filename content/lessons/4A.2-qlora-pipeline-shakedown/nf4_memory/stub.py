BLOCK = 64        # NF4 weights that share one absmax
DQ_BLOCK = 256    # with double quantization: absmaxes that share one fp32 constant


def weight_gb(n_quant, n_bf16, double_quant=True):
    # TODO: bits of absmax each quantized weight pays for. Without double quantization: one fp32
    # absmax per BLOCK weights. With it: one 8-bit absmax per BLOCK, plus one fp32 constant per BLOCK * DQ_BLOCK.
    scale_bits = ...
    # TODO: 4 bits plus scale_bits per quantized weight, 2 bytes per bf16 weight
    total_bytes = ...
    return total_bytes / 1e9


print(f"NF4, no double quant  {weight_gb(n_linear, n_embed, double_quant=False):5.2f} GB")
print(f"NF4 + double quant    {weight_gb(n_linear, n_embed):5.2f} GB")
print(f"bf16, for comparison  {weight_gb(0, n_linear + n_embed):5.2f} GB")
