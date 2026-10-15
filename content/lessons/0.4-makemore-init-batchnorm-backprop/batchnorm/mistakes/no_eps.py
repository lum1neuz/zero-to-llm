import torch

# A batch of 32 pre-activations with 4 features, each on a very different scale.
g = torch.Generator().manual_seed(0)
h = torch.randn((32, 4), generator=g) * torch.tensor([0.5, 2.0, 5.0, 10.0]) + torch.tensor([3.0, -1.0, 0.0, 8.0])
gamma = torch.ones(4)
beta = torch.zeros(4)

def batchnorm(h, gamma, beta, eps=1e-5):
    # TODO: the mean and variance of each column, taken over the batch
    mean = h.mean(0, keepdim=True)
    var = h.var(0, keepdim=True, unbiased=False)
    # TODO: normalize each column to mean 0 and std 1, then scale by gamma and shift by beta
    return gamma * (h - mean) / torch.sqrt(var) + beta


out = batchnorm(h, gamma, beta)
print("mean", out.mean(0))
print("std ", out.std(0, unbiased=False))

bn = torch.nn.BatchNorm1d(4)  # PyTorch's version, in training mode
print("max diff vs nn.BatchNorm1d", (out - bn(h)).abs().max().item())
