
N = torch.tensor([[3, 1, 0, 4],
                  [0, 2, 2, 0],
                  [1, 1, 6, 0],
                  [5, 0, 0, 5]])


def normalize_rows(N):
    # TODO: divide each row by its own total, so every row sums to 1
    return ...


Q = normalize_rows(N)
print(Q)
print(Q.sum(1))
