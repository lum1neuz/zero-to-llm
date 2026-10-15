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
    scale_bits = 8 / BLOCK + 32 / (BLOCK * DQ_BLOCK) if double_quant else 32 / BLOCK
    # TODO: 4 bits plus scale_bits per quantized weight, 2 bytes per bf16 weight
    total_bytes = n_quant * (4 + scale_bits) / 8 + n_bf16 * 2
    return total_bytes / 1e9


print(f"NF4, no double quant  {weight_gb(n_linear, n_embed, double_quant=False):5.2f} GB")
print(f"NF4 + double quant    {weight_gb(n_linear, n_embed):5.2f} GB")
print(f"bf16, for comparison  {weight_gb(0, n_linear + n_embed):5.2f} GB")

def lora_params(shapes, r, layers):
    # TODO: A is [r, in] and B is [out, r] for every adapted Linear, in every layer
    return layers * sum(r * i for i, o in shapes)


def adapter_gb(n_trainable):
    # TODO: per trainable parameter: fp32 weight, fp32 gradient, and Adam's two fp32 moments
    return n_trainable * 16 / 1e9


def logits_gb(seq, vocab, batch):
    # TODO: one fp32 logit per vocabulary entry, for every token in the batch
    return seq * vocab * batch * 4 / 1e9


SEQ = 2048
n_lora = lora_params(SHAPES, r=16, layers=LAYERS)
print(f"LoRA r=16 on all seven Linears: {n_lora / 1e6:.1f}M trainable ({100 * n_lora / (n_linear + n_embed):.2f}%)")
budget = {
    "NF4 base weights": weight_gb(n_linear, n_embed),
    "LoRA weights, grads, Adam": adapter_gb(n_lora),
    "logits (fp32)": logits_gb(SEQ, VOCAB, batch=1),
    "checkpointed activations": LAYERS * SEQ * HIDDEN * 2 / 1e9,
    "one layer recomputed (rough)": 1.6,
    "CUDA context": 0.5,
}
for name, size in budget.items():
    print(f"{name:<30} {size:5.2f} GB")
total = sum(budget.values())
print(f"{'total':<30} {total:5.2f} GB, {'fits' if total < 12 else 'does not fit'} in 12 GB")
