from harness import test, fail, learner_fn, script_error

# Documents that must go, one per rule.
SHORT = "Бүх эрх хуулиар хамгаалагдсан."                                     # 4 words of boilerplate
FIVE = "Өнөөдөр цас их орно гэнэ."                                           # 5 words
JUNK = "1. 2. 3. 4. 5. 6. 7. 8. Үзсэн: 1520 | Сэтгэгдэл: 0"                  # mostly digits and symbols
ENGLISH = "The festival opened on Sukhbaatar Square, which people in the city simply call Төв талбай."  # Latin, with a Mongolian phrase
RUSSIAN = "Сегодня в Улан-Баторе идёт снег, температура опустится до минус двадцати градусов."
KAZAKH = "Бүгін Алматыда қар жауып, ауа температурасы минус жиырма градусқа дейін төмендейді."

# Mongolian documents that must stay, each one near the edge of a rule.
SIX = "Өнөөдөр сургуулийн сурагчид хичээлийн жилээ эхлүүллээ."               # exactly 6 words
DIGITS = "2024 оны 7 сарын 11-нд Үндэсний их баяр наадам өнөөдөр нээлтээ хийлээ."  # digits and spaces, all letters Cyrillic
BRAND = "Монголын үндэсний шигшээ баг Facebook хуудсаараа өнөөдөр шинэ тамирчдаа танилцууллаа."  # one Latin word
ONLY_O = "Монгол Улсын Их Хурлын намрын ээлжит чуулган өнөөдөр нээлтээ хийлээ."  # ө but no ү (the others have both)


def kept(L, text):
    out = learner_fn(L, "keep")(text)
    if out is None:
        fail("`keep` returns nothing for some documents. Every path should end in `return True` or `return False`.")
    if not isinstance(out, bool):
        fail(f"`keep` should return True or False, but it returns {type(out).__name__}.")
    return out


@test("length", "Fragments under 6 words are dropped")
def _(L):
    for text, n in ((SHORT, 4), (FIVE, 5)):
        if kept(L, text):
            fail(f"\"{text}\" is kept, and it's only {n} words. Lines this short are menus, footers and captions. "
                 "Count the words, and drop the document if there are fewer than 6.")


@test("junk", "Documents that are mostly digits and symbols are dropped")
def _(L):
    if kept(L, JUNK):
        fail(f"\"{JUNK}\" is kept. It has 13 words and its few letters are Cyrillic, but only 14 of its 38 non-space characters are letters: "
             "it's a page counter, not text. Compare the number of letters with the number of characters that aren't spaces.")


@test("cyrillic", "Text that isn't mostly Cyrillic is dropped")
def _(L):
    if kept(L, ENGLISH):
        letters = [c for c in ENGLISH if c.isalpha()]
        cyr = sum(0x0400 <= ord(c) <= 0x04FF for c in letters)
        fail(f"\"{ENGLISH}\" is kept. It quotes a Mongolian phrase, so it has an ө, but only {cyr} of its {len(letters)} letters are Cyrillic. "
             "What share of the letters should be Cyrillic for a Mongolian document?")


@test("language", "Russian and Kazakh are dropped")
def _(L):
    if kept(L, RUSSIAN):
        fail(f"The Russian line is kept: \"{RUSSIAN[:40]}…\". It's 100% Cyrillic, so the Cyrillic rule can't catch it. "
             "Which letters does Mongolian use that Russian never does?")
    if kept(L, KAZAKH):
        fail(f"The Kazakh line is kept: \"{KAZAKH[:40]}…\". It has ү, like Mongolian, but also і and қ, which Mongolian never uses. "
             "Drop documents with any letter from `KAZAKH_ONLY`.")


@test("keeps_mongolian", "Every Mongolian document survives")
def _(L):
    controls = {"six": SIX, "digits": DIGITS, "brand": BRAND, "only_o": ONLY_O}
    lost = [k for k, t in controls.items() if not kept(L, t)]
    if not lost:
        return
    if len(lost) == len(controls):
        fail("Your filter drops every Mongolian document. Check each rule's direction: `return False` should run only when a document fails the rule.")
    if "six" in lost:
        fail(f"\"{SIX}\" is dropped, and it has exactly 6 words. \"Fewer than 6\" means 5 or less.")
    if "digits" in lost:
        fail(f"\"{DIGITS}\" is dropped, but every one of its letters is Cyrillic. It also has digits and spaces. "
             "Are you dividing by the number of letters, or by every character?")
    if "only_o" in lost:
        fail(f"\"{ONLY_O[:40]}…\" is dropped. It has ө but no ү, and plenty of Mongolian sentences use only one of them. "
             "Require ө *or* ү, not both.")
    if "brand" in lost:
        letters = [c for c in BRAND if c.isalpha()]
        cyr = sum(0x0400 <= ord(c) <= 0x04FF for c in letters)
        fail(f"\"{BRAND[:44]}…\" is dropped because of the one Latin word, Facebook. {cyr} of its {len(letters)} letters are Cyrillic, {100 * cyr // len(letters)}%. "
             "One brand name shouldn't sink a Mongolian sentence. Is your rule a share, or all-or-nothing?")
    fail(f"\"{controls[lost[0]][:44]}…\" is a clean Mongolian sentence, but `keep` drops it. Check each rule against it.")


@test("script_runs", "Seven of the twelve documents survive")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    docs = L.get("kept")
    if not isinstance(docs, list):
        fail("`kept` is missing. Put back the line `kept = [d for d in DOCS if keep(d)]`.")
    if len(docs) != 7:
        fail(f"{len(docs)} documents survive, but 7 of the 12 are clean Mongolian. Does step 1's `normalize` still pass its tests?")
