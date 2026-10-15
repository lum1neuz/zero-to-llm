
def block_params(d):
    # TODO: count every weight and bias in one Block of width d, without building it
    attn = ...   # c_attn and c_proj
    mlp = ...    # c_fc and c_proj
    norms = ...  # ln_1 and ln_2
    return attn + mlp + norms

for d in (32, 512):
    built = sum(p.numel() for p in Block(d, 4).parameters())
    print(f"d={d:<4} formula {block_params(d):>9,}  built {built:>9,}")
