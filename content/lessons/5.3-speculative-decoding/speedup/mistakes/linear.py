def expected_tokens(alpha, k):
    # TODO: tokens one round yields on average: the drafts that survive, plus the one token the target always adds
    return alpha * k + 1


def speedup(alpha, k, c):
    # TODO: tokens per round over the round's cost in target passes: k draft passes (c each) plus one target pass
    return expected_tokens(alpha, k) / (k * c + 1)


alpha, c = 0.75, 0.15   # a 1B draft for an 8B target
for k in range(1, 9):
    print(f"k={k}   tokens per round {expected_tokens(alpha, k):.2f}   speedup {speedup(alpha, k, c):.2f}x")
