from harness import test, fail, learner_fn, LearnerError, script_error, stdout

BYTES = {i: bytes([i]) for i in range(256)}
CASES = [
    ("ж", True), ("Ж", True), ("ө", True), ("Ү", True), ("ё", True),
    ("a ж", True), ("hello", False), ("", False), ("αβγ", False), ("中文", False), ("�", False),
]


def cyr(L, text):
    out = learner_fn(L, "has_cyrillic")(text)
    if out is None:
        fail("`has_cyrillic` returns None. Its body is still `...`, or a path through it has no `return`.")
    return bool(out)


@test("has_cyrillic", "has_cyrillic finds every Cyrillic letter, Mongolian ones too")
def _(L):
    got = {text: cyr(L, text) for text, _want in CASES}
    wrong = [text for text, want in CASES if got[text] != want]
    if not wrong:
        return
    if got["ж"] and not got["ө"]:
        fail("Your check says 'ө' isn't Cyrillic, but it's a Mongolian letter. Ө, ө, Ү and ү (and Ё, ё) sit outside а…я, "
             "later in Unicode's Cyrillic block, which runs from U+0400 to U+04FF.")
    if got["ж"] and not got["Ж"]:
        fail("Your check misses capitals like 'Ж'. Unicode's Cyrillic block, U+0400 to U+04FF, holds both cases.")
    if got["αβγ"] or got["中文"]:
        fail("Your check says Greek 'αβγ' or Chinese '中文' is Cyrillic. Non-ASCII covers every script. "
             "Look for characters in the Cyrillic block, U+0400 to U+04FF.")
    if got["ж"] and not got["a ж"]:
        fail("'a ж' mixes Latin and Cyrillic, and your check says no. One Cyrillic letter anywhere is enough.")
    if got["hello"] or got[""] or got["�"]:
        fail(f"Your check says {next(t for t in ('hello', '', '�') if got[t])!r} contains Cyrillic, and it doesn't.")
    fail(f"Your check gets {wrong[0]!r} wrong.")


@test("share", "cyrillic_share is the fraction of merged tokens with Cyrillic in them")
def _(L):
    fn = learner_fn(L, "cyrillic_share")
    # Merged tokens: ж (Cyrillic), th, half a letter, ар (Cyrillic): 2 of 4.
    vocab = {**BYTES, 256: "ж".encode("utf-8"), 257: b"th", 258: b"\xd0", 259: "ар".encode("utf-8")}
    try:
        share = fn(vocab)
    except LearnerError as e:
        msg = str(e)
        if "can't decode" in msg or "UnicodeDecodeError" in msg:
            fail("Decoding failed on a merged token that's half a letter, b'\\xd0'. Merged tokens can be pieces of a character, "
                 "so decode with `errors=\"replace\"`, which turns a piece into '�' instead of raising.")
        if "not supported between instances of 'str' and 'int'" in msg or "but int found" in msg:
            fail("`merged` holds bytes, so `has_cyrillic` compares numbers with letters. Decode each token's bytes to text first.")
        raise
    if share is Ellipsis:
        fail("`merged` is still `...`, the placeholder.")
    if abs(share - 0.5) < 1e-9:
        return
    if abs(share - 2 / 260) < 1e-9:
        fail("Your share counts all 260 tokens, the 256 single bytes too. No single byte is a Cyrillic letter, "
             "so they only dilute the share. Count the merged tokens, ids 256 and up.")
    fail(f"With merged tokens ж, th, half a letter and ар, 2 of 4 contain Cyrillic: 0.5. You get {share:g}.")


@test("script_runs", "The script compares both vocabularies")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "contain Cyrillic" not in stdout():
        fail("The comparison loop at the bottom is gone. Put it back to see both vocabularies.")
