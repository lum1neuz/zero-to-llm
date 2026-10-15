import re
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
    words = [w.translate(LOOKALIKES) if has_cyrillic(w) else w for w in re.split(r"\s+", text.strip())]
    return " ".join(words)


DOCS = [normalize(t) for t in RAW]
for raw, doc in zip(RAW, DOCS):
    if raw != doc:
        print("fixed:", doc[:48])
