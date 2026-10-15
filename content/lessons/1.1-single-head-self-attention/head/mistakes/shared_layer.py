import torch

torch.manual_seed(42)
q = torch.randn(5, 4)
k = torch.randn(5, 4)

def causal_weights(q, k):
    T, d = q.shape[-2], q.shape[-1]
    # TODO: score every query against every key, scaled by 1/sqrt(d)
    scores = q @ k.transpose(-2, -1) / d ** 0.5
    # TODO: a T x T grid with 1 where position t may look (keys 0..t) and 0 elsewhere
    tril = torch.tril(torch.ones(T, T))
    # TODO: hide the future, so softmax gives it exactly zero weight
    masked = scores.masked_fill(tril == 0, float("-inf"))
    # TODO: turn each row into weights that sum to 1
    wei = masked.softmax(dim=-1)
    return wei

wei = causal_weights(q, k)
print(wei)
print("row sums", wei.sum(dim=-1))

import time
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        # TODO: three separate projections from n_embd features to head_size, no bias
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = self.key
        self.value = self.key

    def forward(self, x):
        # TODO: project x three ways, then mix the values with causal_weights
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = causal_weights(q, k)
        return wei @ v

torch.manual_seed(1337)
head = Head(32, 16)
x = torch.randn(4, 8, 32)
print("out", tuple(head(x).shape))

with torch.no_grad():
    x256 = torch.randn(1, 256, 32)
    t0 = time.perf_counter()
    for _ in range(20):
        head(x256)
    print(f"T=256 forward: {(time.perf_counter() - t0) / 20 * 1000:.2f} ms")
