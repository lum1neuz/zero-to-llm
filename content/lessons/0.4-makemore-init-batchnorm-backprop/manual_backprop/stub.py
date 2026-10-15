
def manual_backward(X, Y, W1, b1, W2, b2):
    n = X.shape[0]
    hpreact = X @ W1 + b1
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    # loss = F.cross_entropy(logits, Y). Each d<name> is d(loss)/d(<name>), with <name>'s shape.
    # TODO: through cross_entropy
    dlogits = ...
    # TODO: through the second layer, logits = h @ W2 + b2
    dW2 = ...
    db2 = ...
    dh = ...
    # TODO: through tanh, h = tanh(hpreact)
    dhpreact = ...
    # TODO: through the first layer, hpreact = X @ W1 + b1
    dW1 = ...
    db1 = ...
    return {"logits": dlogits, "W2": dW2, "b2": db2, "h": dh, "hpreact": dhpreact, "W1": dW1, "b1": db1}


# Autograd's answer, for comparison.
hpreact = X @ W1 + b1
h = torch.tanh(hpreact)
logits = h @ W2 + b2
for t in (hpreact, h, logits):
    t.retain_grad()
loss = F.cross_entropy(logits, Y)
loss.backward()
auto = {"logits": logits, "W2": W2, "b2": b2, "h": h, "hpreact": hpreact, "W1": W1, "b1": b1}

with torch.no_grad():
    mine = manual_backward(X, Y, W1, b1, W2, b2)
for name, t in auto.items():
    d = mine[name]
    exact = torch.all(d == t.grad).item()
    approx = torch.allclose(d, t.grad)
    maxdiff = (d - t.grad).abs().max().item()
    print(f"{name:8s} | exact: {str(exact):5s} | approximate: {str(approx):5s} | maxdiff: {maxdiff:.1e}")
