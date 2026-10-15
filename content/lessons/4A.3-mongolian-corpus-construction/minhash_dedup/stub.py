
def h(seed, s):
    """A 32-bit hash of s. Each seed gives a different, random-looking order of every possible shingle."""
    x = 2166136261
    for b in seed.to_bytes(4, "little") + s.encode("utf-8"):
        x = ((x ^ b) * 16777619) & 0xFFFFFFFF
    # mix the bits, so that nearby seeds give unrelated orders
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    return x ^ (x >> 16)


def minhash(shingle_set, k=64):
    # TODO: for each seed 0 … k-1, the smallest hash of any shingle in the set
    return ...


def estimate(sig_a, sig_b):
    # TODO: the share of positions where the two signatures agree
    return ...


print(f"docs 0 and 1: exact {jaccard(sets[0], sets[1]):.2f}, MinHash {estimate(minhash(sets[0]), minhash(sets[1])):.2f}")
THRESHOLD = 0.7
sigs, final = [], []
for d in kept:
    sig = minhash(shingles(d))
    if any(estimate(sig, s) >= THRESHOLD for s in sigs):
        print("near-duplicate, dropped:", d[:44])
        continue
    sigs.append(sig)
    final.append(d)
removed = len(kept) - len(final)
print(f"removed {removed} of {len(kept)} as near-duplicates ({100 * removed / len(kept):.1f}%)")
