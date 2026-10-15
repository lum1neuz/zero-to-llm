def mean_nll(probs):
    # TODO: the average negative log-likelihood of these probabilities, in nats
    return ...


# A model that knows nothing gives all 27 symbols the same chance.
n = sum(len(w) + 1 for w in words)  # one prediction per letter, plus the final "."
probs = torch.full((n,), 1 / vocab_size)
print("predictions", n)
print("baseline   ", mean_nll(probs).item())
