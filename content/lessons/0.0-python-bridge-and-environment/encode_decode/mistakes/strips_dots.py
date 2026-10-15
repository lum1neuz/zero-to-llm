# Python warm-up: characters to integers and back

chars = "abcdefghijklmnopqrstuvwxyz"
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0
itos = {i: ch for ch, i in stoi.items()}

def encode(word):
    # TODO: the word with "." at both ends, as a list of integers
    return [stoi[ch] for ch in "." + word + "."]


def decode(ids):
    # TODO: the integers back to characters, joined into one string
    return "".join(itos[i] for i in ids if i != 0)


ids = encode("emma")
print(ids)
print(decode(ids))
