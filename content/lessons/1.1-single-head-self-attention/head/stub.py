
import time
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        # TODO: three separate projections from n_embd features to head_size, no bias
        self.key = ...
        self.query = ...
        self.value = ...

    def forward(self, x):
        # TODO: project x three ways, then mix the values with causal_weights
        k = ...
        q = ...
        v = ...
        wei = causal_weights(q, k)
        return ...

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
