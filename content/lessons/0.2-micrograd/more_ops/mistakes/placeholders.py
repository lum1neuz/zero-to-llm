# micrograd: autograd for single numbers
import math


class Value:
    """A number that remembers how it was made, so gradients can flow back through it."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None  # sends out.grad back to the inputs
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self) = 1 and d(out)/d(other) = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    # Built from + and *, so they need no _backward of their own.
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):  # 2 + v
        return self + other

    def __rmul__(self, other):  # 2 * v
        return self * other

    def backward(self):
        # Step 2 writes topo_order. Until then, call each _backward() by hand.
        self.grad = 1.0
        for v in reversed(topo_order(self)):
            v._backward()

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # TODO: the local derivatives of out = self * other
            d_self = other.data
            d_other = self.data
            self.grad += d_self * out.grad
            other.grad += d_other * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        # TODO: the forward pass, tanh of self.data
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # TODO: the local derivative of tanh, written with t
            local = 1 - t**2
            self.grad += local * out.grad
        out._backward = _backward
        return out


# One neuron, o = tanh(x*w + b). Backprop by hand, from the output back.
x = Value(2.0)
w = Value(-0.5)
b = Value(0.75)
xw = x * w
n = xw + b
o = n.tanh()

o.grad = 1.0
o._backward()
n._backward()
xw._backward()
print("o", o)
print("x", x, " w", w, " b", b)


def topo_order(root):
    # TODO: every Value reachable from root, each one listed after all the Values it was built from
    order = []
    visited = set()

    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            order.append(v)
    build(root)
    return order


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


# Two more ops, attached to Value from out here so you don't have to scroll back up.
def value_pow(self, k):
    assert isinstance(k, (int, float)), "only a constant power"
    out = Value(self.data**k, (self,), f"**{k}")

    def _backward():
        # TODO: the local derivative of x**k
        d_pow = ...
        self.grad += d_pow * out.grad
    out._backward = _backward
    return out


def value_exp(self):
    out = Value(math.exp(self.data), (self,), "exp")

    def _backward():
        # TODO: the local derivative of e**x, written with out.data
        d_exp = ...
        self.grad += d_exp * out.grad
    out._backward = _backward
    return out


Value.__pow__ = value_pow
Value.exp = value_exp
Value.__truediv__ = lambda self, other: self * other**-1


# tanh rebuilt from exp, ** and /, checked against PyTorch.
def tanh_from_exp(n):
    e = (2 * n).exp()
    return (e - 1) / (e + 1)


x, w, b = Value(2.0), Value(-0.5), Value(0.75)
o = tanh_from_exp(x * w + b)
o.backward()
mine = [x.grad, w.grad, b.grad]
ref = torch_grads(2.0, -0.5, 0.75)
print("o        ", o.data)
print("micrograd", [round(g, 6) for g in mine])
print("torch    ", [round(g, 6) for g in ref])
print("max diff ", max(abs(m - r) for m, r in zip(mine, ref)))
