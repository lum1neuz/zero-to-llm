

def batch_inputs(mask):
    # mask: [B, T] bool, True on real tokens. Returns pos [B, T] and keep [B, T, T] for the model.
    B, T = mask.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))  # query t sees keys 0..t
    # TODO: each row's first real token sits at position 0 and they count up from there. Pads can be 0.
    pos = ...
    # TODO: query t may attend to key s when s <= t AND key s is a real token. Shape [B, T, T].
    keep = ...
    return pos, keep


def generate_batched(model, idx, mask, max_new_tokens):
    for _ in range(max_new_tokens):
        pos, keep = batch_inputs(mask)
        next_ids = model(idx, pos, keep)[:, -1].argmax(dim=-1)  # one new id per row
        idx = torch.cat([idx, next_ids[:, None]], dim=1)
        mask = torch.cat([mask, torch.ones(len(mask), 1, dtype=torch.bool)], dim=1)
    return idx, mask


out, out_mask = generate_batched(model, idx, mask, 20)
for b, p in enumerate(prompts):
    alone = generate(model, p[None], 20)[0]
    batched = out[b][out_mask[b]]
    print(f"{decode(batched)!r:32} same as alone: {torch.equal(batched, alone)}")
