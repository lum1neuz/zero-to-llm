def lora_params(shapes, r, layers):
    # TODO: A is [r, in] and B is [out, r] for every adapted Linear, in every layer
    return ...


def adapter_gb(n_trainable):
    # TODO: per trainable parameter: fp32 weight, fp32 gradient, and Adam's two fp32 moments
    return ...


def logits_gb(seq, vocab, batch):
    # TODO: one fp32 logit per vocabulary entry, for every token in the batch
    return ...


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
