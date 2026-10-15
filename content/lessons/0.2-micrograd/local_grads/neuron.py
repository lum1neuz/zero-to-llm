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
