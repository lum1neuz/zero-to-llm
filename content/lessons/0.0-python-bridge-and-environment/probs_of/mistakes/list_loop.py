import torch

P = torch.tensor([[0.1, 0.6, 0.3],
                  [0.5, 0.25, 0.25],
                  [0.2, 0.2, 0.6]])

xs = torch.tensor([0, 2, 1, 0])
ys = torch.tensor([1, 0, 2, 2])

def probs_of(P, xs, ys):
    # TODO: one number per pair, P at row xs[i] and column ys[i]
    return [P[x, y] for x, y in zip(xs, ys)]


print(probs_of(P, xs, ys))
