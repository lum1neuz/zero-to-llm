# A short story to train on. Every token you make comes from these bytes.
text = """Once upon a time, there was a little bird named Tumen. Tumen lived in a tall tree by a quiet river. Every morning, Tumen sang a happy song for the other birds in the tree.
One day, the wind was very strong. The wind blew the nest out of the tree and into the river. Tumen was sad. He did not know what to do.
A kind old turtle saw the nest in the water. "Do not be sad, little bird," said the turtle. "I will help you." The turtle swam to the nest and pushed it to the shore.
Tumen was so happy! He thanked the turtle and carried the nest back up the tree, one stick at a time. That night, Tumen sang his best song for the turtle.
From that day on, the little bird and the old turtle were best friends. They played by the river every day, and when the wind was strong, they helped each other."""

ids = list("aaabdaaabac".encode("utf-8"))
print(ids)

def get_stats(ids):
    # TODO: count how often each pair of neighbours (a, b) appears in ids
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    # TODO: copy ids, writing idx wherever pair appears instead of its two ids
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

stats = get_stats(ids)
top = max(stats, key=stats.get)
print("top pair", top, bytes(top), "seen", stats[top], "times")
print(merge(ids, top, 256))

def train(text, vocab_size):
    ids = list(text.encode("utf-8"))
    merges = {}                                   # (a, b) -> the new id
    vocab = {i: bytes([i]) for i in range(256)}   # id -> the bytes it stands for
    for i in range(vocab_size - 256):
        stats = get_stats(ids)
        # TODO: the most frequent pair becomes the next new id
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx
        # TODO: the new token stands for its two parts' bytes, joined
        vocab[idx] = bytes(pair)
    return merges, vocab

merges, vocab = train(text, 276)
for pair, idx in list(merges.items())[:12]:
    print(idx, pair, vocab[idx])
