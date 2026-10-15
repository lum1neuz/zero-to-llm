import torch
import torch.nn.functional as F

# 81 names, enough to see the idea. On your machine you'll train on all 32,033 in names.txt.
words = """
emma olivia ava isabella sophia charlotte mia amelia harper evelyn abigail emily elizabeth mila
ella avery sofia camila aria scarlett victoria madison luna grace chloe penelope layla riley zoey
nora lily eleanor hannah lillian addison aubrey ellie stella natalie zoe liam noah oliver elijah
william james benjamin lucas henry alexander mason michael ethan daniel jacob logan jackson levi
sebastian mateo jack owen theodore aiden samuel joseph john david wyatt matthew luke asher carter
julian grayson leo jayden gabriel isaac lincoln anthony
""".split()

chars = "abcdefghijklmnopqrstuvwxyz"
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0
itos = {i: ch for ch, i in stoi.items()}

# The training set: every (current character, next character) pair, as integers.
xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
print("pairs", xs.nelement())


def forward(W, xs):
    xenc = F.one_hot(xs, num_classes=27).float()  # [n, 27], one row per input character
    logits = xenc @ W  # [n, 27], read them as log-counts
    # TODO: softmax: exponentiate, then make each row sum to 1
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    return probs


def loss_fn(probs, ys):
    # TODO: the average negative log-probability given to the right next characters
    return F.nll_loss(probs.log(), ys)


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
    W.data += -50 * W.grad
print("final loss", round(loss.item(), 4))
