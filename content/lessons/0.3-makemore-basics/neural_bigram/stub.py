
def forward(W, xs):
    xenc = F.one_hot(xs, num_classes=27).float()  # [n, 27], one row per input character
    logits = xenc @ W  # [n, 27], read them as log-counts
    # TODO: softmax: exponentiate, then make each row sum to 1
    probs = ...
    return probs


def loss_fn(probs, ys):
    # TODO: the average negative log-probability given to the right next characters
    return ...


g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
for step in range(100):
    probs = forward(W, xs)
    loss = loss_fn(probs, ys)
    if step % 25 == 0:
        print("step", step, "loss", round(loss.item(), 4))
    W.grad = None
    loss.backward()
    # TODO: one step of gradient descent on W, learning rate 50
    ...
print("final loss", round(loss.item(), 4))
