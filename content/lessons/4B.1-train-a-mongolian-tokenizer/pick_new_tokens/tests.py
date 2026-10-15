from harness import test, fail, learner_fn, script_error, stdout

LETTERS = "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
# A base vocab with a few two-letter pieces, and corpus counts built so that every wrong way of
# scoring picks a different top 3:
#   token      count  base pieces          saved = count × (pieces − 1)
#   хүүхэд       21   х ү ү х э д   (6)    105
#   байна        30   б ай н а      (4)     90
#   ын           80   ы н           (2)     80
#   андаг        35   ан да г       (3)     70
#   даг          55   да г          (2)     55
#   ан, э       100, 200  already base tokens
BASE = set(LETTERS) | {"ан", "ай", "да", "аг"}
COUNTS = {"ан": 100, "э": 200, "ын": 80, "даг": 55, "байна": 30, "хүүхэд": 21, "андаг": 35}


def encode(word, vocab):
    toks, i = [], 0
    while i < len(word):
        j = len(word)
        while j > i + 1 and word[i:j] not in vocab:
            j -= 1
        toks.append(word[i:j])
        i = j
    return toks


def ranked(score, counts, base, k, keep=lambda t, base: t not in base, reverse=True):
    cands = [t for t in counts if keep(t, base)]
    s = {t: score(t, counts, base) for t in cands}
    return sorted(s, key=lambda t: (-s[t], t) if reverse else (s[t], t))[:k]


def saved(t, counts, base):
    return counts[t] * (len(encode(t, base)) - 1)


def fmt(toks):
    return "[" + ", ".join(f'"{t}"' for t in toks) + "]"


def call(L, counts, base, k):
    fn = learner_fn(L, "pick_new_tokens")
    out = fn(dict(counts), set(base), k)
    if out is None:
        fail("`pick_new_tokens` returns nothing. It has to return the list of chosen tokens.")
    if out is Ellipsis:
        fail("`pick_new_tokens` still returns `...`, the placeholder. Fill in `candidates`, `saved` and the return line.")
    if isinstance(out, dict):
        fail("`pick_new_tokens` returns a dict of scores. Return a list of the tokens themselves, best saver first.")
    if isinstance(out, (set, frozenset)):
        fail("`pick_new_tokens` returns a set, which has no order. Return a list, best saver first.")
    out = list(out) if isinstance(out, tuple) else out
    if not isinstance(out, list):
        fail(f"`pick_new_tokens` should return a list of tokens, but it returns {type(out).__name__}.")
    if out and isinstance(out[0], (tuple, list)):
        fail("`pick_new_tokens` returns (token, score) pairs. Return just the tokens, so the script can add them to a vocab.")
    if not all(isinstance(t, str) for t in out):
        fail(f"`pick_new_tokens` should return tokens (strings), but it returned {out[:3]}.")
    if len(out) != k:
        fail(f"The tests asked for {k} tokens and got {len(out)}. Keep only the first k of the ranking.")
    return out


@test("returns_tokens", "pick_new_tokens returns a list of k tokens")
def _(L):
    call(L, COUNTS, BASE, 3)


@test("best_savers", "The k tokens that save the most base tokens, best first")
def _(L):
    for k in (3, 2):
        got = call(L, COUNTS, BASE, k)
        want = ranked(saved, COUNTS, BASE, k)
        if got == want:
            continue
        stories = [
            (ranked(lambda t, c, b: c[t], COUNTS, BASE, k),
             f"Your picks, {fmt(got)}, are the candidates ranked by how often they appear. But \"ын\" (80 uses) is only 2 base pieces, "
             f"so each use saves 1 token, while \"хүүхэд\" (21 uses) is 6 pieces and saves 5. What does each use of a token save?"),
            (ranked(lambda t, c, b: c[t], COUNTS, BASE, k, keep=lambda t, b: True),
             f"Your picks, {fmt(got)}, include \"э\" and \"ан\", which are already in the base vocab, so adding them saves nothing. "
             "Rank by tokens saved, and leave out what the base vocab already has."),
            (ranked(lambda t, c, b: c[t] * (len(t) - 1), COUNTS, BASE, k),
             f"Your picks, {fmt(got)}, score each token by its letters. But the base vocab doesn't spend one token per letter: "
             "\"андаг\" has 5 letters and only 3 base pieces (ан, да, г). Count the pieces `encode(t, base_vocab)` makes."),
            (ranked(lambda t, c, b: c[t] * len(encode(t, b)), COUNTS, BASE, k),
             f"Your picks, {fmt(got)}, score each token by all of its base pieces. But the new token costs one token itself, so "
             "\"ын\" (2 pieces) saves 1 per use, not 2. Subtract that one."),
            (ranked(saved, COUNTS, BASE, k, reverse=False),
             f"Your picks, {fmt(got)}, are the smallest savers first. Sort from the biggest saving down."),
        ]
        g = L.get("BASE_VOCAB")
        if isinstance(g, (set, frozenset)):
            stories.append((ranked(saved, COUNTS, g, k),
                            "`pick_new_tokens` ignores the `base_vocab` it's given and uses the script's `BASE_VOCAB`. "
                            "On another base vocab it picks the wrong tokens. Use the argument."))
        for wrong, text in stories:
            if got == wrong:
                fail(text)
        if sorted(got) == sorted(want):
            fail(f"You picked the right tokens in the wrong order: {fmt(got)}. Put the biggest saver first, {fmt(want)}.")
        fail(f"With these counts the {k} biggest savers are {fmt(want)}, but you picked {fmt(got)}. "
             "Score each candidate by `counts[t]` × (base pieces − 1).")


@test("script_runs", "The script runs and prints the three vocab sizes")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "new" not in L:
        fail("The loop that calls `pick_new_tokens` for k = 5, 10 and 20 is missing. Put it back at the bottom.")
    if len(stdout().strip().splitlines()) < 5:
        fail("Some of the print lines are missing. Put them back so you can compare the vocab sizes.")
