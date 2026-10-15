
def step0_loss(E, H, ids):
    # TODO: next-token logits at every position except the last
    logits = ...
    # TODO: cross-entropy against the token that really comes next
    return ...

base_ids = to_ids(HELD_OUT, BASE)
ext_ids = to_ids(HELD_OUT, EXT)
g = torch.Generator().manual_seed(0)
n, d = len(NEW_TOKENS), E.shape[1]
E_rand = torch.cat([E, torch.randn(n, d, generator=g)])
H_rand = torch.cat([H, torch.randn(n, d, generator=g)])
print(f"base vocab, base model         {step0_loss(E, H, base_ids).item():.2f} nats per token ({len(base_ids)} tokens)")
print(f"uniform guess over {len(EXT)} tokens  {math.log(len(EXT)):.2f}")
print(f"extended, random new rows      {step0_loss(E_rand, H_rand, ext_ids).item():.2f} nats per token ({len(ext_ids)} tokens)")
print(f"extended, mean of pieces       {step0_loss(E_ext, H_ext, ext_ids).item():.2f}")
