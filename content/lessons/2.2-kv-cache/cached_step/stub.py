def step(model, token, pos, cache):
    # Runs ONE new token at position pos. cache["k"] and cache["v"] hold one row per earlier token.
    x = model.embed(token, pos)                          # [C]
    q, k, v = x @ model.Wq, x @ model.Wk, x @ model.Wv   # each [C]
    # TODO: add this token's k and v to the cache, as new last rows
    cache["k"] = ...
    cache["v"] = ...
    # TODO: q's attention weights over every cached key: scaled dot products, then softmax.
    # No mask needed: the cache holds only this token and the ones before it.
    att = ...
    # TODO: mix the cached values with those weights
    h = ...
    return model.head(token, h)                          # logits [V]


cache = {"k": torch.zeros(0, model.C), "v": torch.zeros(0, model.C)}
full = model(prompt)  # 2.1's way: every position at once
for pos in range(len(prompt)):
    logits = step(model, prompt[pos], pos, cache)
    diff = (logits - full[pos]).abs().max().item()
    print(f"pos {pos}  cache rows {cache['k'].shape[0]}  max diff vs full {diff:.1e}")
