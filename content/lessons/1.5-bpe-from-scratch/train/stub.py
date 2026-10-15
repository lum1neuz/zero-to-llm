
def train(text, vocab_size):
    ids = list(text.encode("utf-8"))
    merges = {}                                   # (a, b) -> the new id
    vocab = {i: bytes([i]) for i in range(256)}   # id -> the bytes it stands for
    for i in range(vocab_size - 256):
        stats = get_stats(ids)
        # TODO: the most frequent pair becomes the next new id
        pair = ...
        idx = ...
        ids = merge(ids, pair, idx)
        merges[pair] = idx
        # TODO: the new token stands for its two parts' bytes, joined
        vocab[idx] = ...
    return merges, vocab

merges, vocab = train(text, 276)
for pair, idx in list(merges.items())[:12]:
    print(idx, pair, vocab[idx])
