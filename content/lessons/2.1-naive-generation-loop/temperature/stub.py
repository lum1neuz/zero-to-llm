

def sample(logits, temperature=1.0, generator=None):
    # TODO: divide by the temperature, then turn the logits into probabilities
    probs = ...
    # TODO: draw one id from probs with torch.multinomial, using the generator
    return ...


g = torch.Generator().manual_seed(42)
for t in (0.5, 1.0, 1.5):
    out = generate(model, prompt, 60, pick=lambda logits: sample(logits, t, g))
    print(f"T={t}: {decode(out)!r}")
