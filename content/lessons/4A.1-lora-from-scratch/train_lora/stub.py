def trainable(model):
    # TODO: a list of the parameters that still require grad
    return ...


def train_step(model, opt, X, Y):
    # TODO: clear the gradients from the last step
    ...
    # TODO: mean squared error between the model's output and Y
    loss = ...
    # TODO: backpropagate, then let the optimizer update A and B
    ...
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
