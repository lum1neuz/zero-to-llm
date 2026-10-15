def expected_tokens(alpha, k):
    # TODO: tokens one round yields on average: the drafts that survive, plus the one token the target always adds
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def speedup(alpha, k, c):
    # TODO: tokens per round over the round's cost in target passes: k draft passes (c each) plus one target pass
    return expected_tokens(alpha, k) / (k * c + 1)


