# VRAM budget for a QLoRA fine-tune of Llama 3.1 8B on a 12 GB GPU

# Llama 3.1 8B
LAYERS = 32
HIDDEN, INTER, KV = 4096, 14336, 1024
VOCAB = 128256

# (in, out) of the seven Linear layers in every block
SHAPES = [(HIDDEN, HIDDEN), (HIDDEN, KV), (HIDDEN, KV), (HIDDEN, HIDDEN),   # q, k, v, o
          (HIDDEN, INTER), (HIDDEN, INTER), (INTER, HIDDEN)]                # gate, up, down
n_linear = LAYERS * sum(i * o for i, o in SHAPES)   # quantized to NF4
n_embed = 2 * VOCAB * HIDDEN                        # embed_tokens and lm_head stay bf16
print(f"{(n_linear + n_embed) / 1e9:.2f}B parameters, {n_linear / 1e9:.2f}B of them in Linear layers")

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
