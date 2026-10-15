
def is_partial(piece):
    # TODO: True when these bytes aren't whole characters by themselves. Can Python decode them?
    ...

def partial_share(sentences, merges, vocab):
    pieces = [vocab[i] for s in sentences for i in encode(s, merges)]
    # TODO: the fraction of pieces that are only part of a character
    return ...

for name, merges, vocab in (("English BPE", en_merges, en_vocab), ("Mongolian BPE", mn_merges, mn_vocab)):
    print(f"{name:<14} {100 * partial_share(mongolian_test, merges, vocab):5.1f}% of tokens are pieces of a letter")
print("Өнөөдөр →", [mn_vocab[i].decode("utf-8", errors="replace") for i in encode("Өнөөдөр", mn_merges)])
