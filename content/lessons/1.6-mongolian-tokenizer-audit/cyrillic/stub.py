
def has_cyrillic(text):
    # TODO: True if any character is a Cyrillic letter, Mongolian's Ө, ө, Ү and ү included
    ...

def cyrillic_share(vocab):
    # TODO: the text of every merged token (ids 256 and up). Some are half a letter, so don't let decoding fail
    merged = ...
    return sum(has_cyrillic(t) for t in merged) / len(merged)

for name, vocab in (("English BPE", en_vocab), ("Mongolian BPE", mn_vocab)):
    print(f"{name:<14} {100 * cyrillic_share(vocab):5.1f}% of merged tokens contain Cyrillic")
