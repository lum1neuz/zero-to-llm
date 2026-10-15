def speculative_greedy(prompt, n_new, k):
    tokens = list(prompt)
    rounds = 0
    while len(tokens) < len(prompt) + n_new and rounds < n_new:   # each round should add at least one token
        rounds += 1
        # TODO: the draft proposes k tokens, one at a time, each one seeing the ones before it
        proposal = ...
        # TODO: ONE target pass over tokens + proposal scores every drafted position at once
        logits = ...
        # TODO: row n - 1 + i is the target's own choice for position n + i. Append the target's choice
        # and stop at the first disagreement. If all k agree, the last row gives one more token for free.
        ...
    return tokens[:len(prompt) + n_new]


prompt = [3, 1]
calls["target"] = 0
expected = greedy(prompt, 24)
plain = calls["target"]
calls["target"] = 0
got = speculative_greedy(prompt, 24, k=4)
print("same tokens as greedy:", got == expected)
print(f"target passes for 24 tokens: greedy {plain}, speculative {calls['target']}")
