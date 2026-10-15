# A tiny two-layer MLP: 8 examples, 6 inputs, 10 hidden tanh units, 5 classes.
g = torch.Generator().manual_seed(42)
n, d_in, d_hidden, vocab = 8, 6, 10, 5
X = torch.randn((n, d_in), generator=g)
Y = torch.randint(0, vocab, (n,), generator=g)
W1 = (torch.randn((d_in, d_hidden), generator=g) * 0.5).requires_grad_()
b1 = (torch.randn(d_hidden, generator=g) * 0.1).requires_grad_()
W2 = (torch.randn((d_hidden, vocab), generator=g) * 0.5).requires_grad_()
b2 = (torch.randn(vocab, generator=g) * 0.1).requires_grad_()

