def causal_weights(q, k):
    T, d = q.shape[-2], q.shape[-1]
    # TODO: score every query against every key, scaled by 1/sqrt(d)
    scores = ...
    # TODO: a T x T grid with 1 where position t may look (keys 0..t) and 0 elsewhere
    tril = ...
    # TODO: hide the future, so softmax gives it exactly zero weight
    masked = ...
    # TODO: turn each row into weights that sum to 1
    wei = ...
    return wei

wei = causal_weights(q, k)
print(wei)
print("row sums", wei.sum(dim=-1))
