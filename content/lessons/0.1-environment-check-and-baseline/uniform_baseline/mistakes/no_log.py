import torch

words = ["emma", "olivia", "ava", "isabella", "sophia"]
vocab_size = 27  # a to z, plus "." for the start and end of a name

def mean_nll(probs):
    # TODO: the average negative log-likelihood of these probabilities, in nats
    return -probs.mean()


# A model that knows nothing gives all 27 symbols the same chance.
n = sum(len(w) + 1 for w in words)  # one prediction per letter, plus the final "."
probs = torch.full((n,), 1 / vocab_size)
print("predictions", n)
print("baseline   ", mean_nll(probs).item())
