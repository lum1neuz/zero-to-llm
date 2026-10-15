# Python warm-up: characters to integers and back

chars = "abcdefghijklmnopqrstuvwxyz"
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0
itos = {i: ch for ch, i in stoi.items()}

def encode(word):
    # TODO: the word with "." at both ends, as a list of integers
    return [0] + [chars.index(ch) for ch in word] + [0]


def decode(ids):
    # TODO: the integers back to characters, joined into one string
    return "".join(itos[i] for i in ids)


ids = encode("emma")
print(ids)
print(decode(ids))
