V = 64  # a cut-down vocabulary, so the browser can hold the logits


def stand_in(text, chars_per_token, nll_per_token, seed):
    """Token ids for text, and logits that give each next token about exp(-nll_per_token) probability.
    Stand-in numbers shaped like a real eval's: position t scores the token at t + 1."""
    g = torch.Generator().manual_seed(seed)
    n = round(len(text) / chars_per_token) + 1                # + 1 for the BOS token at the start
    ids = torch.randint(1, V, (n,), generator=g)
    ids[0] = 0                                                # BOS
    logits = torch.randn(n, V, generator=g) * 0.3
    p = math.exp(-nll_per_token)
    logits[torch.arange(n - 1), ids[1:]] += math.log((V - 1) * p / (1 - p))
    return text, logits, ids


HELDOUT = [
    "Улаанбаатар хотод өнөөдөр цас орж, агаарын температур хасах хорин хэм хүрнэ.",
    "Малчид өвөлжөөндөө бэлтгэлээ базааж байна.",
    "Наадмын баяр долдугаар сарын арван нэгнээс эхэлж, гурван өдөр үргэлжилнэ.",
]
# your control run: Llama 3.1's tokenizer cuts Mongolian into short pieces
CONTROL = [stand_in(t, 2.4, 1.15, seed) for seed, t in enumerate(HELDOUT)]
# a preview of 4B: a Mongolian tokenizer, longer tokens, same text
PREVIEW_4B = [stand_in(t, 5.6, 2.6, 10 + seed) for seed, t in enumerate(HELDOUT)]

