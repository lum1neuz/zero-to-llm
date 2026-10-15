

def kv_cache_bytes(cfg, ctx, bytes_per_value=2):
    # TODO: the width of one attention head
    head_dim = ...
    # TODO: K and V, in every layer, for every KV head, at every position
    return ...


llama31_8b = {"num_hidden_layers": 32, "num_attention_heads": 32,
              "num_key_value_heads": 8, "hidden_size": 4096}
gib = kv_cache_bytes(llama31_8b, 8192) / 2**30
print(f"Llama 3.1 8B at 8k context: {gib:.2f} GiB of KV cache per sequence")
