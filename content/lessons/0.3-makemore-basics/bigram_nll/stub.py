

def nll(P, words):
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            # TODO: the log of the probability P gives to ch2 coming after ch1
            logprob = ...
            log_likelihood += logprob
            n += 1
    # TODO: the average negative log-likelihood
    return ...


print("bigram loss", nll(P, words).item())
print("uniform    ", torch.tensor(27.0).log().item())
