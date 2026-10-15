import torch
import torch.nn.functional as F

# 2.1's tiny model, now batched: ids [B, T] in, logits [B, T, V] out.
# pos says where each token sits, and keep says which keys each query may attend to.
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

    def __call__(self, idx, pos=None, keep=None):
        B, T = idx.shape
        if pos is None:   # no padding: every row counts 0, 1, 2, ...
            pos = torch.arange(T).expand(B, T)
        if keep is None:  # no padding: plain causal mask
            keep = torch.tril(torch.ones(T, T, dtype=torch.bool)).expand(B, T, T)
        x = F.embedding(idx, self.wte) + F.embedding(pos, self.wpe)
        q, k, v = x @ self.Wq, x @ self.Wk, x @ self.Wv
        att = (q @ k.transpose(1, 2)) / self.C**0.5
        # The most negative float, not -inf: a row with no allowed keys gets junk instead of NaN
        att = att.masked_fill(~keep, torch.finfo(att.dtype).min)
        h = F.softmax(att, dim=-1) @ v
        return self.bigram[idx] + h @ self.Wo


def generate(model, idx, max_new_tokens):
    # 2.1's greedy loop, for one unpadded sequence of shape [1, T]
    for _ in range(max_new_tokens):
        next_id = model(idx)[:, -1].argmax(dim=-1)
        idx = torch.cat([idx, next_id[:, None]], dim=1)
    return idx


model = TinyLM()

prompts = [encode("the cat"), encode("a dog"), encode("the")]

def left_pad(prompts, pad_id=0):
    T = max(len(p) for p in prompts)
    idx = torch.full((len(prompts), T), pad_id)
    mask = torch.zeros(len(prompts), T, dtype=torch.bool)  # True on real tokens
    for b, p in enumerate(prompts):
        # TODO: write prompt p into row b of idx, flush against the right edge
        idx[b, T - len(p):] = p
        # TODO: mark those same columns as real tokens in mask
        mask[b, T - len(p):] = True
    return idx, mask


idx, mask = left_pad(prompts)
print(idx)
print(mask)


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
