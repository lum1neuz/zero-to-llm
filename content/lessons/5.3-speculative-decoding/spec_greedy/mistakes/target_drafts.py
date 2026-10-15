import torch

V = 8
g = torch.Generator().manual_seed(0)
T = torch.randn(V, V, generator=g) * 2          # target: next-token logits after each token
D = T + torch.randn(V, V, generator=g)          # draft: the same table, blurred

calls = {"target": 0, "draft": 0}

def target(tokens):
    # one forward pass: next-token logits after every position, shape [len(tokens), V]
    calls["target"] += 1
    return T[tokens]

def draft(tokens):
    calls["draft"] += 1
    return D[tokens]

def greedy(prompt, n_new):
    tokens = list(prompt)
    for _ in range(n_new):
        tokens.append(int(target(tokens)[-1].argmax()))
    return tokens

def speculative_greedy(prompt, n_new, k):
    tokens = list(prompt)
    rounds = 0
    while len(tokens) < len(prompt) + n_new and rounds < n_new:   # each round should add at least one token
        rounds += 1
        # TODO: the draft proposes k tokens, one at a time, each one seeing the ones before it
        proposal = []
        for _ in range(k):
            proposal.append(int(target(tokens + proposal)[-1].argmax()))
        # TODO: ONE target pass over tokens + proposal scores every drafted position at once
        logits = target(tokens + proposal)
        # TODO: row n - 1 + i is the target's own choice for position n + i. Append the target's choice
        # and stop at the first disagreement. If all k agree, the last row gives one more token for free.
        n = len(tokens)
        for i in range(k + 1):
            choice = int(logits[n - 1 + i].argmax())
            tokens.append(choice)
            if i == k or choice != proposal[i]:
                break
    return tokens[:len(prompt) + n_new]


prompt = [3, 1]
calls["target"] = 0
expected = greedy(prompt, 24)
plain = calls["target"]
calls["target"] = 0
got = speculative_greedy(prompt, 24, k=4)
print("same tokens as greedy:", got == expected)
print(f"target passes for 24 tokens: greedy {plain}, speculative {calls['target']}")
