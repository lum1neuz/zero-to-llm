def count_bigrams(words):
    N = torch.zeros((27, 27), dtype=torch.int64)
    for w in words:
        # TODO: the characters of w, with "." added at both ends
        chs = ...
        for ch1, ch2 in zip(chs, chs[1:]):
            # TODO: add one to the cell for this pair
            ...
    return N


N = count_bigrams(words)
print("pairs counted", N.sum().item())
print("most common first letter:", itos[N[0].argmax().item()])
