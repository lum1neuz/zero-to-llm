def left_pad(prompts, pad_id=0):
    T = max(len(p) for p in prompts)
    idx = torch.full((len(prompts), T), pad_id)
    mask = torch.zeros(len(prompts), T, dtype=torch.bool)  # True on real tokens
    for b, p in enumerate(prompts):
        # TODO: write prompt p into row b of idx, flush against the right edge
        ...
        # TODO: mark those same columns as real tokens in mask
        ...
    return idx, mask


idx, mask = left_pad(prompts)
print(idx)
print(mask)
