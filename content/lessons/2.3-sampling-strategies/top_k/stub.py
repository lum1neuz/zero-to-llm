def top_k(logits, k):
    # TODO: the k-th largest logit. Anything smaller gets cut.
    kth = ...
    # TODO: every logit below kth becomes -inf, so softmax gives it probability 0
    return ...


row = torch.tensor([2.0, -1.0, 0.5, 3.0, 0.2, 1.0])
print("top_k(row, 3):", top_k(row, 3))
print("probs:        ", F.softmax(top_k(row, 3), dim=-1))
