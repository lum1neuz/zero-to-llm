

def distinct_2(ids):
    # TODO: every pair of neighbouring ids, as tuples
    pairs = ...
    # TODO: the fraction of those pairs that are unique
    return ...


strategies = {
    "greedy": greedy,
    "T=1.0": lambda logits: sample(logits, 1.0, g),
    "top-k 5": lambda logits: sample(logits, 1.0, g, keep=lambda x: top_k(x, 5)),
    "top-p 0.9": lambda logits: sample(logits, 1.0, g, keep=lambda x: top_p(x, 0.9)),
}
for name, pick in strategies.items():
    g = torch.Generator().manual_seed(0)
    out = generate(model, prompt, 60, pick=pick)
    print(f"{name:>9}  distinct-2 {distinct_2(out.tolist()):.2f}  {decode(out)!r}")
