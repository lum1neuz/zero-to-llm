import math
import torch
import torch.nn.functional as F

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

def doc_nll(logits, ids):
    # TODO: summed negative log-likelihood, in nats, of every next token. Position t predicts ids[t + 1].
    return F.cross_entropy(logits[:-1], ids[1:], reduction="sum").item()


def heldout(docs):
    # TODO: pool every predicted token of every document, then turn the total into perplexity and bits per character
    total_nll = sum(doc_nll(logits, ids) for _, logits, ids in docs)
    n_tokens = sum(len(ids) - 1 for _, _, ids in docs)
    n_chars = sum(len(text) for text, _, _ in docs)
    ppl = math.exp(sum(doc_nll(logits, ids) / (len(ids) - 1) for _, logits, ids in docs) / len(docs))
    bpc = total_nll / math.log(2) / n_chars
    return ppl, bpc


for name, docs in (("control, Llama tokenizer   ", CONTROL), ("4B preview, Mongolian tok.", PREVIEW_4B)):
    ppl, bpc = heldout(docs)
    print(f"{name}  ppl {ppl:6.2f}   bits/char {bpc:.3f}")
