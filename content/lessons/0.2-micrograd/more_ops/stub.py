

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
