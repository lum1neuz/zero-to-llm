    def tanh(self):
        # TODO: the forward pass, tanh of self.data
        t = ...
        out = Value(t, (self,), "tanh")

        def _backward():
            # TODO: the local derivative of tanh, written with t
            local = ...
            self.grad += local * out.grad
        out._backward = _backward
        return out


