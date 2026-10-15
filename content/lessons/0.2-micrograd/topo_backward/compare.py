# The same neuron, checked against PyTorch's autograd.
import torch


def micrograd_grads(xv, wv, bv):
    x, w, b = Value(xv), Value(wv), Value(bv)
    o = (x * w + b).tanh()
    o.backward()
    return [x.grad, w.grad, b.grad]


def torch_grads(xv, wv, bv):
    x, w, b = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (xv, wv, bv))
    o = torch.tanh(x * w + b)
    o.backward()
    return [x.grad.item(), w.grad.item(), b.grad.item()]


mine = micrograd_grads(2.0, -0.5, 0.75)
ref = torch_grads(2.0, -0.5, 0.75)
print("micrograd", [round(g, 6) for g in mine])
print("torch    ", [round(g, 6) for g in ref])
print("max diff ", max(abs(m - r) for m, r in zip(mine, ref)))
