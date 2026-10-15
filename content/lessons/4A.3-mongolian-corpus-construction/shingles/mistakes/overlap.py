import unicodedata

# Twelve documents as they came off the web, artifacts and all
RAW = [
    "Улаанбаатaр хотoд өнөөдөр цас орж, агаарын температур хасах 20 хэм хүрнэ гэж цаг уурын албанаас мэдээллээ.",
    "Улаанбаатар хотод  өнөөдөр цас орж, агаарын температур хасах 20 хэм хүрнэ гэж цаг уурын албанаас мэдээллээ. Эх сурвалж: МОНЦАМЭ",
    "Монгол Улсын Их Хурлын намрын ээлжит чуулган өнөөдөр нээлтээ хийлээ.",
    "Монгол Улсын Их Хурлын намрын ээлжит чуулган өнөөдөр нээлтээ хийлээ. Чуулганаар ирэх оны төсвийн төслийг хэлэлцэнэ.",
    "Малчид  өвөлжөөндөө\u00a0бэлтгэлээ\tбазааж, өвс тэжээлээ нөөцөлж байна.",
    "Сургуулии\u0306н сурагчид шинэ хичээлии\u0306н жилээ эхлүүллээ.",
    "2024 оны 7 сарын 11-нд Үндэсний их баяр наадам нээлтээ хийлээ.",
    "Сегодня в Улан-Баторе идёт снег, температура опустится до минус двадцати градусов.",
    "Бүгін Алматыда қар жауып, ауа температурасы минус жиырма градусқа дейін төмендейді.",
    "Click here to subscribe to our newsletter and get the latest news.",
    "Бүх эрх хуулиар хамгаалагдсан.",
    "1. 2. 3. 4. 5. 6. 7. 8. Үзсэн: 1520 | Сэтгэгдэл: 0",
]

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
    text = unicodedata.normalize("NFC", text)
    # TODO: split on any run of whitespace, and fix look-alikes, but only in words that contain Cyrillic
    words = [w.translate(LOOKALIKES) if has_cyrillic(w) else w for w in text.split()]
    return " ".join(words)


DOCS = [normalize(t) for t in RAW]
for raw, doc in zip(RAW, DOCS):
    if raw != doc:
        print("fixed:", doc[:48])

MONGOLIAN_ONLY = set("өүӨҮ")           # letters Mongolian has and Russian doesn't
KAZAKH_ONLY = set("әғқңұһіӘҒҚҢҰҺІ")    # letters Kazakh has and Mongolian doesn't


def is_cyrillic(ch):
    return "\u0400" <= ch <= "\u04ff"


def keep(text):
    words = text.split()
    letters = [c for c in text if c.isalpha()]
    # TODO: drop fragments shorter than 6 words
    if len(words) < 6:
        return False
    # TODO: drop junk: letters must be at least half of the characters that aren't spaces
    if len(letters) < 0.5 * len(text.replace(" ", "")):
        return False
    # TODO: drop text that isn't Cyrillic: at least 80% of the letters must be Cyrillic
    if sum(is_cyrillic(c) for c in letters) < 0.8 * len(letters):
        return False
    # TODO: Russian is Cyrillic too. Keep only text with ө or ү in it, and none of the letters only Kazakh uses
    if not any(c in MONGOLIAN_ONLY for c in letters) or any(c in KAZAKH_ONLY for c in letters):
        return False
    return True


kept = [d for d in DOCS if keep(d)]
for d in DOCS:
    print("keep" if keep(d) else "drop", "|", d[:44])
print(f"kept {len(kept)} of {len(DOCS)}")

def shingles(text, n=3):
    # TODO: every run of n consecutive words, each joined into one string with single spaces, as a set
    words = text.split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a, b):
    # TODO: the shingles a and b share, divided by all the distinct shingles between them
    return len(a & b) / min(len(a), len(b))


sets = [shingles(d) for d in kept]
for i in range(len(kept)):
    for j in range(i + 1, len(kept)):
        J = jaccard(sets[i], sets[j])
        if J > 0.2:
            print(f"docs {i} and {j}: Jaccard {J:.2f}")
