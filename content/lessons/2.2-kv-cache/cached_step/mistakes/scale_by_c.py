import torch
import torch.nn.functional as F

# The tiny model from 2.1, split into pieces you can run one token at a time.
# embed() adds token and position embeddings; head() turns the attention output into logits.
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
        self.positions = 0  # how many positions embed() has processed, in total

    def embed(self, idx, pos):
        self.positions += torch.as_tensor(idx).numel()
        return self.wte[idx] + self.wpe[pos]

    def head(self, idx, h):
        return self.bigram[idx] + h @ self.Wo

    def __call__(self, idx):
        # 2.1's forward: every position at once, with a causal mask
        T = idx.shape[0]
        x = self.embed(idx, torch.arange(T))
        q, k, v = x @ self.Wq, x @ self.Wk, x @ self.Wv
        att = (q @ k.T) / self.C**0.5
        att = att.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf"))
        return self.head(idx, F.softmax(att, dim=-1) @ v)


def generate(model, idx, max_new_tokens):
    # 2.1's greedy loop: re-runs the model on the whole sequence every step
    for _ in range(max_new_tokens):
        next_id = model(idx)[-1].argmax()
        idx = torch.cat([idx, next_id.view(1)])
    return idx


model = TinyLM()
prompt = encode("the ")

def step(model, token, pos, cache):
    # Runs ONE new token at position pos. cache["k"] and cache["v"] hold one row per earlier token.
    x = model.embed(token, pos)                          # [C]
    q, k, v = x @ model.Wq, x @ model.Wk, x @ model.Wv   # each [C]
    # TODO: add this token's k and v to the cache, as new last rows
    cache["k"] = torch.cat([cache["k"], k[None]])
    cache["v"] = torch.cat([cache["v"], v[None]])
    # TODO: q's attention weights over every cached key: scaled dot products, then softmax.
    # No mask needed: the cache holds only this token and the ones before it.
    att = F.softmax(cache["k"] @ q / model.C, dim=-1)
    # TODO: mix the cached values with those weights
    h = att @ cache["v"]
    return model.head(token, h)                          # logits [V]


cache = {"k": torch.zeros(0, model.C), "v": torch.zeros(0, model.C)}
full = model(prompt)  # 2.1's way: every position at once
for pos in range(len(prompt)):
    logits = step(model, prompt[pos], pos, cache)
    diff = (logits - full[pos]).abs().max().item()
    print(f"pos {pos}  cache rows {cache['k'].shape[0]}  max diff vs full {diff:.1e}")
