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
            local = 1 - self.data**2
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
