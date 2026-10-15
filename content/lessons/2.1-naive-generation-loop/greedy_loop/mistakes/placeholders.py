import torch
import torch.nn.functional as F

# A tiny character-level model with fixed weights: a bigram table counted from one line of
# text, plus one attention head, so each next character depends on the whole context.
# It stands in for your 30M model: ids [T] in, logits [T, V] out.
text = "the cat sat on the mat. the dog sat on the log. a cat and a dog ran to the mat. "
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: torch.tensor([stoi[c] for c in s])
decode = lambda ids: "".join(chars[i] for i in ids.tolist())


class TinyLM:
    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        V, C, CTX = len(chars), 16, 64
        self.C = C
        counts = torch.ones(V, V) * 0.1
        for a, b in zip(text, text[1:]):
            counts[stoi[a], stoi[b]] += 1
        self.bigram = (counts / counts.sum(dim=1, keepdim=True)).log()
        self.wte = torch.randn(V, C, generator=g)
        self.wpe = torch.randn(CTX, C, generator=g) * 0.5
        self.Wq = torch.randn(C, C, generator=g) / C**0.5
        self.Wk = torch.randn(C, C, generator=g) / C**0.5
        self.Wv = torch.randn(C, C, generator=g) / C**0.5
        self.Wo = torch.randn(C, V, generator=g) * 0.15
        self.positions = 0  # how many positions the model has processed, in total

    def __call__(self, idx):
        T = idx.shape[0]
        self.positions += T
        x = self.wte[idx] + self.wpe[:T]
        q, k, v = x @ self.Wq, x @ self.Wk, x @ self.Wv
        att = (q @ k.T) / self.C**0.5
        att = att.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf"))
        h = F.softmax(att, dim=-1) @ v
        return self.bigram[idx] + h @ self.Wo


model = TinyLM()

prompt = encode("the ")

def greedy(logits):
    # TODO: the id of the largest logit in this one row
    return ...


def generate(model, idx, max_new_tokens, pick=greedy):
    for _ in range(max_new_tokens):
        # TODO: run the model on every token so far
        logits = ...
        # TODO: pick the next id from the LAST position's row of logits
        next_id = ...
        # TODO: append it to the end of idx
        idx = ...
    return idx


out = generate(model, prompt, 60)
print(repr(decode(out)))
print("positions processed:", model.positions)
