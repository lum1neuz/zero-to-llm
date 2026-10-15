import math
import torch
import torch.nn.functional as F

torch.manual_seed(42)
q = torch.randn(5, 4)
k = torch.randn(5, 4)

def causal_weights(q, k):
    T, d = q.shape[-2], q.shape[-1]
    # TODO: score every query against every key, scaled by 1/sqrt(d)
    scores = q @ k.mT / math.sqrt(d)
    # TODO: a T x T grid with 1 where position t may look (keys 0..t) and 0 elsewhere
    tril = torch.tril(torch.ones(T, T))
    # TODO: hide the future, so softmax gives it exactly zero weight
    masked = scores.masked_fill(tril == 0, float("-inf"))
    # TODO: turn each row into weights that sum to 1
    wei = F.softmax(masked, dim=-1)
    return wei

wei = causal_weights(q, k)
print(wei)
print("row sums", wei.sum(dim=-1))
