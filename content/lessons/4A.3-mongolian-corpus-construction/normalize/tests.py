import unicodedata
from harness import test, fail, learner_fn, LearnerError, script_error

BREVE = chr(0x0306)      # combining breve: и + BREVE displays as й
DIAER = chr(0x0308)      # combining diaeresis: o + DIAER displays as ö
NBSP = chr(0x00A0)       # non-breaking space


def show(s):
    """Spell out invisible characters so a diagnostic can point at them."""
    names = {BREVE: "[breve]", DIAER: "[diaeresis]", NBSP: "[nbsp]", "\t": "[tab]"}
    return "".join(names.get(c, c) for c in s)


def norm(L, text):
    fn = learner_fn(L, "normalize")
    try:
        out = fn(text)
    except LearnerError as e:
        if "can only join an iterable" in str(e):
            fail("`words` is still `...`, the placeholder, so there's nothing to join. It should be the list of fixed words.")
        raise
    if not isinstance(out, str):
        fail(f"`normalize` should return a string, but it returns {type(out).__name__}.")
    return out


@test("composes", "Letters typed in pieces become single characters")
def _(L):
    raw = "Сургуулии" + BREVE + "н сурагчид"
    want = "Сургуулийн сурагчид"
    out = norm(L, raw)
    if out == want:
        composed = norm(L, want)
        if composed != want:
            fail(f"Already-composed text comes back changed: {show(composed)}. Normalizing should leave clean text alone.")
        return
    if BREVE in out:
        if BREVE in norm(L, want):
            fail("Your output splits й into и plus a combining breve, two code points. That's NFD, which takes letters apart. "
                 "You want the form that puts them together.")
        fail(f"й arrived as и + a combining breve, two code points that look like one letter, and it's still split: {show(out)}. "
             "A tokenizer sees a different string from the й typed on a Mongolian keyboard. Which Unicode normal form composes them?")
    fail(f"`normalize` turns {show(raw)} into {show(out)}, but it should give {want}.")


@test("whitespace", "Runs of spaces, tabs and non-breaking spaces become one space")
def _(L):
    raw = "Малчид  өвөлжөөндөө" + NBSP + "бэлтгэлээ\tбазааж байна"
    want = "Малчид өвөлжөөндөө бэлтгэлээ базааж байна"
    out = norm(L, raw)
    if out == want:
        return
    if "  " in out:
        fail(f"Two spaces in a row survive: {show(out)}. `split(\" \")` makes an empty word between them. What does `split()` with no argument do?")
    if NBSP in out or "\t" in out:
        fail(f"A tab or a non-breaking space survives: {show(out)}. Splitting on \" \" only breaks at plain spaces. "
             "Which `split` breaks at any whitespace?")
    fail(f"`normalize` turns {show(raw)} into {show(out)}, but it should give {want}.")


@test("lookalikes", "Latin look-alikes inside Cyrillic words become Cyrillic")
def _(L):
    raw = "Улаанбаатaр хотoд цас орно"
    want = "Улаанбаатар хотод цас орно"
    out = norm(L, raw)
    if out == want:
        return
    if out == raw:
        fail("The a in Улаанбаатaр and the o in хотoд are Latin (U+0061, U+006F), not Cyrillic. They look identical, "
             "but to a tokenizer and to dedup they're different words. Run the Cyrillic words through `LOOKALIKES`.")
    fail(f"`normalize` turns {raw} into {out}, but it should give {want}.")


@test("latin_words_kept", "Words with no Cyrillic stay as they are")
def _(L):
    raw = "Facebook хуудсаар дамжуулан Coca-Cola"
    out = norm(L, raw)
    if out == raw:
        return
    if out.split()[0] != "Facebook" or out.split()[-1] != "Coca-Cola":
        fail(f"Facebook and Coca-Cola came back as {out.split()[0]} and {out.split()[-1]}: their Latin letters were swapped for Cyrillic ones. "
             "A Latin word quoted inside Mongolian text should stay Latin. Only fix words that already contain Cyrillic.")
    fail(f"`normalize` changed {raw} into {out}. Words without Cyrillic should come back unchanged.")


@test("order", "Composing happens before the look-alike fix")
def _(L):
    raw = "o" + DIAER + "нөөдөр"
    want = "өнөөдөр"
    out = norm(L, raw)
    if out == want:
        return
    if out.startswith(chr(0x04E7)):
        fail("ö arrived as Latin o plus a combining diaeresis. You swapped the o for Cyrillic о first, so composing then made "
             "ӧ (Cyrillic o with diaeresis, a letter from other Cyrillic alphabets), not ө. Compose first, so the whole ö meets `LOOKALIKES`.")
    fail(f"`normalize` turns {show(raw)} into {show(out)}, but it should give {want}.")


@test("script_runs", "Your script normalizes all twelve documents")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    docs = L.get("DOCS")
    if not isinstance(docs, list) or len(docs) != 12:
        fail("`DOCS` should hold the twelve normalized documents. Put the line `DOCS = [normalize(t) for t in RAW]` back.")
