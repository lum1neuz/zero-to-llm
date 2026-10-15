import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
base = nn.Linear(6, 10)
x = torch.randn(3, 6)

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        # TODO: freeze every parameter of the base layer
        self.base.requires_grad_(False)
        # TODO: A is [r, in] with small random values, B is [out, r] of zeros. Both train.
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        # TODO: how much the update counts, from alpha and r
        self.scale = alpha / r

    def forward(self, x):
        # TODO: the frozen layer's output plus the scaled low-rank update
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


lora = LoRALinear(base, r=4, alpha=8)
print("same as base at step 0:", torch.equal(lora(x), base(x)))
print("trainable:", [name for name, p in lora.named_parameters() if p.requires_grad])

torch.manual_seed(1)
X = torch.randn(64, 6)
delta = torch.randn(10, 2) @ torch.randn(2, 6) * 0.5
Y = base(X) + X @ delta.T

def trainable(model):
    # TODO: a list of the parameters that still require grad
    return [model.B]


def train_step(model, opt, X, Y):
    # TODO: clear the gradients from the last step
    opt.zero_grad()
    # TODO: mean squared error between the model's output and Y
    loss = F.mse_loss(model(X), Y)
    # TODO: backpropagate, then let the optimizer update A and B
    loss.backward()
    opt.step()
    return loss.item()


model = LoRALinear(base, r=4, alpha=8)
params = trainable(model)
n_train = sum(p.numel() for p in params)
n_all = sum(p.numel() for p in model.parameters())
print(f"trainable {n_train} of {n_all} parameters ({100 * n_train / n_all:.1f}%)")

opt = torch.optim.Adam(params, lr=0.05)
w0 = base.weight.clone()
for step in range(201):
    loss = train_step(model, opt, X, Y)
    if step % 50 == 0:
        print(f"step {step:>3}  loss {loss:.5f}")
print("base weights unchanged:", torch.equal(base.weight, w0))
