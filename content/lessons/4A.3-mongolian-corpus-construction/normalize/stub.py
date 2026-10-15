# Latin and other look-alikes that turn up inside Cyrillic words
LOOKALIKES = str.maketrans({
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у",
    "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М", "H": "Н", "O": "О", "P": "Р", "C": "С", "T": "Т", "X": "Х",
    "ö": "ө", "ü": "ү", "Ö": "Ө", "Ü": "Ү", "ѳ": "ө", "Ѳ": "Ө",
})


def has_cyrillic(word):
    return any("\u0400" <= ch <= "\u04ff" for ch in word)


def normalize(text):
    # TODO: compose letters that arrived in pieces (и + a combining breve → й) into single characters
    text = ...
    # TODO: split on any run of whitespace, and fix look-alikes, but only in words that contain Cyrillic
    words = ...
    return " ".join(words)


DOCS = [normalize(t) for t in RAW]
for raw, doc in zip(RAW, DOCS):
    if raw != doc:
        print("fixed:", doc[:48])
