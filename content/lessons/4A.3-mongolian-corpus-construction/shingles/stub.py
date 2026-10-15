
def shingles(text, n=3):
    # TODO: every run of n consecutive words, each joined into one string with single spaces, as a set
    words = text.split()
    return ...


def jaccard(a, b):
    # TODO: the shingles a and b share, divided by all the distinct shingles between them
    return ...


sets = [shingles(d) for d in kept]
for i in range(len(kept)):
    for j in range(i + 1, len(kept)):
        J = jaccard(sets[i], sets[j])
        if J > 0.2:
            print(f"docs {i} and {j}: Jaccard {J:.2f}")
