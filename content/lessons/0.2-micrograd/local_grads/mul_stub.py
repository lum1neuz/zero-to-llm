    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # TODO: the local derivatives of out = self * other
            d_self = ...
            d_other = ...
            self.grad += d_self * out.grad
            other.grad += d_other * out.grad
        out._backward = _backward
        return out

