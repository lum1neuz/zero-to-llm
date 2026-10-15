# A batch of 32 pre-activations with 4 features, each on a very different scale.
g = torch.Generator().manual_seed(0)
h = torch.randn((32, 4), generator=g) * torch.tensor([0.5, 2.0, 5.0, 10.0]) + torch.tensor([3.0, -1.0, 0.0, 8.0])
gamma = torch.ones(4)
beta = torch.zeros(4)

