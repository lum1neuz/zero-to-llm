import torch

P = torch.tensor([[0.1, 0.6, 0.3],
                  [0.5, 0.25, 0.25],
                  [0.2, 0.2, 0.6]])

xs = torch.tensor([0, 2, 1, 0])
ys = torch.tensor([1, 0, 2, 2])

def probs_of(P, xs, ys):
    # TODO: one number per pair, P at row xs[i] and column ys[i]
    return P[xs, ys]


print(probs_of(P, xs, ys))

N = torch.tensor([[3, 1, 0, 4],
                  [0, 2, 2, 0],
                  [1, 1, 6, 0],
                  [5, 0, 0, 5]])


def normalize_rows(N):
    # TODO: divide each row by its own total, so every row sums to 1
    return N / N.sum(1).unsqueeze(1)


Q = normalize_rows(N)
print(Q)
print(Q.sum(1))
