import re
from harness import test, fail, learner_fn, script_error, stdout

LETTERS = "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
# Two sentences with capitals, punctuation and a line break, so splitting the text the wrong way
# changes the count.
TEXT = ("Хүүхдүүд сургууль руу явж байна.\nБид гэртээ хоол идэж байна. "
        "Ээж маань бууз хийж байна, хүүхдүүд хоол идэж байна.")
# A vocab that is neither the script's BASE_VOCAB nor its MN_VOCAB, so code that reads a global
# vocab instead of its argument gets a different number.
VOCAB_A = set(LETTERS) | {"бай", "на", "хүүхд", "үүд", "гэр", "тээ", "хоол", "сур", "гууль", "яв"}
VOCAB_B = set(LETTERS) | {"байна", "хүүхдүүд", "сургууль", "бид"}


def words(text):
    return re.findall(r"\w+", text.lower())


def encode(word, vocab):
    toks, i = [], 0
    while i < len(word):
        j = len(word)
        while j > i + 1 and word[i:j] not in vocab:
            j -= 1
        toks.append(word[i:j])
        i = j
    return toks


def tokens_of(ws, vocab):
    return [t for w in ws for t in encode(w, vocab)]


def call(L, text, vocab):
    fn = learner_fn(L, "fertility")
    out = fn(text, vocab)
    if out is None:
        fail("`fertility` returns nothing. It has to end with `return` and the number of tokens per word.")
    if out is Ellipsis:
        fail("`fertility` still returns `...`, the placeholder. Fill in `ws`, `n_tokens` and the return line.")
    if isinstance(out, tuple):
        fail(f"`fertility` returns {len(out)} values, `{out}`. It should return one number: tokens divided by words.")
    if isinstance(out, bool) or not isinstance(out, (int, float)):
        fail(f"`fertility` should return a number, but it returns {type(out).__name__}.")
    return float(out)


def story(got, text, vocab, L):
    """Name the mistake behind a wrong fertility, from the number alone."""
    ws = words(text)
    toks = tokens_of(ws, vocab)
    n, t = len(ws), len(toks)
    near = lambda a, b: abs(a - b) < 1e-6
    if near(got, sum(len(w) for w in ws) / n):
        return (f"You got {got:.2f}, the number of letters per word: you counted letters, not tokens. "
                "`encode(w, vocab)` turns a word into tokens. How many does each word become?")
    if near(got, n / t):
        return (f"You got {got:.2f}, which is below 1: that's words per token, the inverse. "
                "Fertility is tokens per word. Which count goes on top?")
    if near(got, len(set(toks)) / n):
        return (f"You got {got:.2f}: you counted distinct tokens, so a token used twice counts once. "
                "Fertility counts every token the text costs, repeats included.")
    if near(got, len(encode(text, vocab)) / n) or near(got, len(encode(text.lower(), vocab)) / n):
        return ("You encoded the whole text as one string, so spaces and newlines became tokens, and capitals and "
                "punctuation stayed in. Encode each word from `words(text)` separately.")
    sw = text.split()
    for ws2 in (sw, [w.lower() for w in sw]):
        for denom in (len(ws2), n):
            if near(got, len(tokens_of(ws2, vocab)) / denom):
                return ("Your words come from `text.split()`, so \"байна.\" keeps its full stop and \"Бид\" its capital, "
                        "and both cost extra tokens. Use `words(text)`, which splits the text the same way the corpus was split.")
    if near(got, t // n):
        return f"You got {got:.0f}, a whole number: `//` throws away the fraction. Use `/`."
    return f"On a test sentence of {n} words that encodes to {t} tokens, `fertility` gives {got:.4f} instead of {t / n:.4f}."


@test("returns_number", "fertility returns one number")
def _(L):
    call(L, TEXT, VOCAB_A)


@test("tokens_per_word", "fertility is total tokens / words, for the vocab it's given")
def _(L):
    ws = words(TEXT)
    vocabs = (VOCAB_A, VOCAB_B)
    got = [call(L, TEXT, v) for v in vocabs]
    want = [len(tokens_of(ws, v)) / len(ws) for v in vocabs]
    if all(abs(g - w) <= 1e-6 for g, w in zip(got, want)):
        return
    # Two vocabs that should give different numbers gave the same one, a global vocab's number.
    used = [name for name in ("BASE_VOCAB", "MN_VOCAB")
            if isinstance(L.get(name), (set, frozenset)) and abs(got[0] - len(tokens_of(ws, L[name])) / len(ws)) < 1e-6]
    if abs(got[0] - got[1]) < 1e-9 and used:
        fail(f"Two different vocabs should give {want[0]:.2f} and {want[1]:.2f} tokens per word, but `fertility` "
             f"ignores the `vocab` it's given and encodes with `{used[0]}` instead. Pass `vocab` to `encode`.")
    for g, w, v in zip(got, want, vocabs):
        if abs(g - w) > 1e-6:
            fail(story(g, TEXT, v, L))


@test("script_runs", "The script runs and prints both tokenizers")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "MN_VOCAB" not in L:
        fail("The lines that train `MN_VOCAB` and print both fertilities are missing. Put them back at the bottom.")
    if len(stdout().strip().splitlines()) < 2:
        fail("The two print lines are missing, so there's nothing to compare. Add them back.")
