
MONGOLIAN_ONLY = set("өүӨҮ")           # letters Mongolian has and Russian doesn't
KAZAKH_ONLY = set("әғқңұһіӘҒҚҢҰҺІ")    # letters Kazakh has and Mongolian doesn't


def is_cyrillic(ch):
    return "\u0400" <= ch <= "\u04ff"


def keep(text):
    words = text.split()
    letters = [c for c in text if c.isalpha()]
    # TODO: drop fragments shorter than 6 words
    ...
    # TODO: drop junk: letters must be at least half of the characters that aren't spaces
    ...
    # TODO: drop text that isn't Cyrillic: at least 80% of the letters must be Cyrillic
    ...
    # TODO: Russian is Cyrillic too. Keep only text with ө or ү in it, and none of the letters only Kazakh uses
    ...
    return True


kept = [d for d in DOCS if keep(d)]
for d in DOCS:
    print("keep" if keep(d) else "drop", "|", d[:44])
print(f"kept {len(kept)} of {len(DOCS)}")
