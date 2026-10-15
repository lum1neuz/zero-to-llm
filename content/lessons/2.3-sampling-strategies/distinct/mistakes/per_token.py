import torch
import torch.nn.functional as F

# The tiny model from 2.1: a bigram table counted from one line of text plus one attention head.
# ids [T] in, logits [T, V] out.
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
    return logits.argmax()


def sample(logits, temperature=1.0, generator=None, keep=None):
    # 2.1's temperature sampling, plus an optional filter that cuts unlikely tokens first
    logits = logits / temperature
    if keep is not None:
        logits = keep(logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator)


def generate(model, idx, max_new_tokens, pick=greedy):
    for _ in range(max_new_tokens):
        next_id = pick(model(idx)[-1])
        idx = torch.cat([idx, next_id.view(1)])
    return idx

def top_k(logits, k):
    # TODO: the k-th largest logit. Anything smaller gets cut.
    kth = logits.topk(k).values[-1]
    # TODO: every logit below kth becomes -inf, so softmax gives it probability 0
    return logits.masked_fill(logits < kth, float("-inf"))


row = torch.tensor([2.0, -1.0, 0.5, 3.0, 0.2, 1.0])
print("top_k(row, 3):", top_k(row, 3))
print("probs:        ", F.softmax(top_k(row, 3), dim=-1))


def top_p(logits, p):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, order = probs.sort(descending=True)
    # TODO: the running total of the sorted probabilities
    cum = sorted_probs.cumsum(dim=-1)
    # TODO: True for every token to cut: the tokens ranked above it already cover p
    cut_sorted = cum - sorted_probs >= p
    cut = cut_sorted[order.argsort()]  # back to the vocabulary's order
    return logits.masked_fill(cut, float("-inf"))


for p in (0.5, 0.9, 0.99):
    kept = (top_p(row, p) > float("-inf")).sum().item()
    print(f"top_p(row, {p}) keeps {kept} of {len(row)} tokens")


def distinct_2(ids):
    # TODO: every pair of neighbouring ids, as tuples
    pairs = list(zip(ids, ids[1:]))
    # TODO: the fraction of those pairs that are unique
    return len(set(pairs)) / len(ids)


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
