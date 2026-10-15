import torch

def linear_weights(fan_in, fan_out, g, gain):
    # TODO: a [fan_in, fan_out] matrix of normal random weights, drawn with g, with std gain / sqrt(fan_in)
    return torch.rand((fan_in, fan_out), generator=g) * gain / fan_in**0.5


# Five tanh layers, 100 wide. Watch the spread of the activations and how many are saturated.
g = torch.Generator().manual_seed(42)
x = torch.randn((500, 100), generator=g)
h = x
for layer in range(5):
    W = linear_weights(100, 100, g, gain=5 / 3)
    h = torch.tanh(h @ W)
    saturated = (h.abs() > 0.97).float().mean().item()
    print(f"layer {layer}  std {h.std().item():.2f}  saturated {saturated:.0%}")
