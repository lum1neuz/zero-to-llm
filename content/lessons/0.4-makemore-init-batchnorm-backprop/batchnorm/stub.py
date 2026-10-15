def batchnorm(h, gamma, beta, eps=1e-5):
    # TODO: the mean and variance of each column, taken over the batch
    mean = ...
    var = ...
    # TODO: normalize each column to mean 0 and std 1, then scale by gamma and shift by beta
    return ...


out = batchnorm(h, gamma, beta)
print("mean", out.mean(0))
print("std ", out.std(0, unbiased=False))

bn = torch.nn.BatchNorm1d(4)  # PyTorch's version, in training mode
print("max diff vs nn.BatchNorm1d", (out - bn(h)).abs().max().item())
