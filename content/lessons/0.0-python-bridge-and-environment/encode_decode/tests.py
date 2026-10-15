from harness import test, fail, learner_fn

LETTERS = "abcdefghijklmnopqrstuvwxyz"
STOI = {ch: i + 1 for i, ch in enumerate(LETTERS)}
STOI["."] = 0


def expected(word):
    return [STOI[ch] for ch in "." + word + "."]


def call(L, name, *args):
    """Call the learner's function. Returns (result, exception) so a test can diagnose the exception."""
    learner_fn(L, name)  # explains a missing or misnamed function
    try:
        return L[name](*args), None
    except Exception as exc:  # noqa: BLE001 - anything the learner's code raises
        return None, exc


def encoded(L, word):
    out, exc = call(L, "encode", word)
    if exc is not None:
        learner_fn(L, "encode")(word)  # re-raise, explained in plain words
    if out is Ellipsis:
        fail("`encode` still returns `...`, the placeholder. Build the list of integers for the word.")
    if out is None:
        fail("`encode` returns nothing. Did the `return` go missing?")
    if isinstance(out, str):
        fail(f"`encode({word!r})` returns the string {out!r}. The model needs numbers: a list with one integer per character.")
    if type(out).__name__ == "generator":
        fail("`encode` returns a generator, not a list. Parentheses around a comprehension make a lazy generator. Which brackets make a list?")
    if not isinstance(out, list):
        fail(f"`encode` should return a list, but it returns {type(out).__name__}.")
    return out


@test("encode_list", "encode returns a list of integers")
def _(L):
    out = encoded(L, "ava")
    if not all(isinstance(i, int) for i in out):
        kinds = sorted({type(i).__name__ for i in out})
        fail(f"`encode('ava')` holds {', '.join(kinds)} values. Every item should be a plain int from `stoi`.")


@test("encode_values", "encode wraps the word in \".\" and uses stoi")
def _(L):
    out = encoded(L, "ava")
    want = expected("ava")
    if out == want:
        return
    if out == want[1:-1]:
        fail(f"`encode('ava')` gives {out}, with no 0 at either end. "
             "Where does the model see that a name starts, and where it stops?")
    if out == want[:-1] or out == want[1:]:
        fail(f"`encode('ava')` gives {out}, with a 0 at only one end. A name needs \".\" before its first letter *and* after its last.")
    if out[1:-1] == [ord(ch) for ch in "ava"]:
        fail(f"`encode('ava')` gives {out}. Those are Unicode code points from `ord()`. The model's vocabulary is the 27 entries in `stoi`.")
    if out[1:-1] == [STOI[ch] - 1 for ch in "ava"]:
        fail(f"`encode('ava')` gives {out}, so 'a' comes out as 0, the same number as \".\". "
             "The model couldn't tell \"a\" from the end of a name. Which table already reserves 0 for \".\"?")
    fail(f"`encode('ava')` gives {out}, but it should be {want}: 0 for \".\", then 'a' = 1, 'v' = 22, 'a' = 1, then 0.")


@test("decode_string", "decode turns integers back into one string")
def _(L):
    ids = expected("emma")
    out, exc = call(L, "decode", ids)
    if isinstance(exc, KeyError):
        fail(f"`decode` raised KeyError: {exc}. It looked an integer up in a table whose keys are characters. "
             "Which of your two tables goes from integers to characters?")
    if exc is not None:
        learner_fn(L, "decode")(ids)
    if out is Ellipsis:
        fail("`decode` still returns `...`, the placeholder.")
    if isinstance(out, list):
        fail(f"`decode` returns the list {out[:4]}…, one character per item. "
             "`\"\".join(...)` glues a sequence of strings into one.")
    if not isinstance(out, str):
        fail(f"`decode` should return a string, but it returns {type(out).__name__}.")
    if out == "emma":
        fail("`decode` returns 'emma' and drops the 0s. Keep it a plain inverse of `encode`: every integer becomes a character, so you get '.emma.'.")
    if out != ".emma.":
        fail(f"`decode({ids})` gives {out!r}, but it should be '.emma.'.")


@test("round_trip", "decode(encode(w)) gives the word back")
def _(L):
    for word in ("zoe", "bartholomew", "q"):
        ids = encoded(L, word)
        out = learner_fn(L, "decode")(ids)
        if out != "." + word + ".":
            fail(f"`decode(encode({word!r}))` gives {out!r}, not {'.' + word + '.'!r}.")
