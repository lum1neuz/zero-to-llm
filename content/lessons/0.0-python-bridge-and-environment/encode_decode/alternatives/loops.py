# Python warm-up: characters to integers and back

chars = "abcdefghijklmnopqrstuvwxyz"
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0
itos = {i: ch for ch, i in stoi.items()}

def encode(word):
    out = []
    for ch in "." + word + ".":
        out.append(stoi[ch])
    return out


def decode(ids):
    text = ""
    for i in ids:
        text += itos[i]
    return text


ids = encode("emma")
print(ids)
print(decode(ids))
