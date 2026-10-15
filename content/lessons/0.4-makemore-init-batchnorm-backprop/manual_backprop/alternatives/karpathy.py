import torch
import torch.nn.functional as F

# A tiny two-layer MLP: 8 examples, 6 inputs, 10 hidden tanh units, 5 classes.
g = torch.Generator().manual_seed(42)
n, d_in, d_hidden, vocab = 8, 6, 10, 5
X = torch.randn((n, d_in), generator=g)
Y = torch.randint(0, vocab, (n,), generator=g)
W1 = (torch.randn((d_in, d_hidden), generator=g) * 0.5).requires_grad_()
b1 = (torch.randn(d_hidden, generator=g) * 0.1).requires_grad_()
W2 = (torch.randn((d_hidden, vocab), generator=g) * 0.5).requires_grad_()
b2 = (torch.randn(vocab, generator=g) * 0.1).requires_grad_()


def manual_backward(X, Y, W1, b1, W2, b2):
    n = X.shape[0]
    hpreact = X @ W1 + b1
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    # loss = F.cross_entropy(logits, Y). Each d<name> is d(loss)/d(<name>), with <name>'s shape.
    # TODO: through cross_entropy
    dlogits = F.softmax(logits, 1)
    dlogits[range(n), Y] -= 1
    dlogits /= n
    # TODO: through the second layer, logits = h @ W2 + b2
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0, keepdim=True)
    dh = dlogits @ W2.T
    # TODO: through tanh, h = tanh(hpreact)
    dhpreact = (1 - h**2) * dh
    # TODO: through the first layer, hpreact = X @ W1 + b1
    dW1 = X.T @ dhpreact
    db1 = dhpreact.sum(0, keepdim=True)
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
