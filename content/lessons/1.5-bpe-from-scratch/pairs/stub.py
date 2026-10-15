def get_stats(ids):
    # TODO: count how often each pair of neighbours (a, b) appears in ids
    counts = {}
    for pair in ...:
        counts[pair] = ...
    return counts

def merge(ids, pair, idx):
    # TODO: copy ids, writing idx wherever pair appears instead of its two ids
    out = []
    i = 0
    while i < len(ids):
        if ...:
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
