from harness import test, fail, warn, learner_fn, script_error, stdout

BYTES = {i: bytes([i]) for i in range(256)}
CASES = [
    (b"a", False, "b'a' (the letter a, one byte)"),
    ("а".encode("utf-8"), False, "b'\\xd0\\xb0' (the Cyrillic letter а, two bytes)"),
    ("ө".encode("utf-8"), False, "b'\\xd3\\xa9' (the Mongolian letter ө)"),
    (b"\xd0", True, "b'\\xd0' (the first half of a Cyrillic letter)"),
    (b"\xb0", True, "b'\\xb0' (the second half of а)"),
    ("аб".encode("utf-8") + b"\xd0", True, "b'\\xd0\\xb0\\xd0\\xb1\\xd0' (\"аб\" plus half a letter)"),
]


def partial(L, piece):
    out = learner_fn(L, "is_partial")(piece)
    if out is None:
        fail("`is_partial` returns None. Its body is still `...`, or a path through it has no `return`.")
    if out is Ellipsis:
        fail("`is_partial` returns `...`, the placeholder.")
    return bool(out)


@test("is_partial", "is_partial spots bytes that aren't whole characters")
def _(L):
    got = [(piece, want, what, partial(L, piece)) for piece, want, what in CASES]
    wrong = [(p, w, what, g) for p, w, what, g in got if g != w]
    if not wrong:
        return
    if all(g != w for _p, w, _what, g in got):
        fail("Every answer is backwards: whole letters come out partial and half-letters come out whole. Swap your True and False.")
    piece, want, what, g = wrong[0]
    if not want and piece == b"a":
        fail("You call b'a' partial, but it's the letter a, a whole character in one byte. "
             "Length alone doesn't tell you. Ask Python whether the bytes decode.")
    if not want:
        fail(f"You call {what} partial, but it's a whole character. A letter can take several bytes. Ask Python whether the bytes decode.")
    if all(not g for _p, _w, _what, g in got):
        fail("Nothing comes out partial, not even b'\\xd0' by itself. If you decode with `errors=\"replace\"`, decoding never fails, "
             "so it can't tell you anything. Let it raise, and catch `UnicodeDecodeError`.")
    fail(f"You call {what} whole, but it can't be decoded on its own.")


@test("share", "partial_share is the fraction of tokens that are pieces of a letter")
def _(L):
    fn = learner_fn(L, "partial_share")
    # "а б" with "а" merged: tokens а, space, \xd0, \xb1, so 2 of 4 are partial.
    out = fn(["а б"], {(208, 176): 256}, {**BYTES, 256: "а".encode("utf-8")})
    if out is Ellipsis:
        fail("`partial_share` still returns `...`, the placeholder.")
    try:
        share = float(out)
    except (TypeError, ValueError):
        fail(f"`partial_share` should return a number, but it returns {type(out).__name__}.")
    if abs(share - 0.5) < 1e-9:
        return
    if abs(share - 2) < 1e-9:
        fail("You return 2, the number of partial tokens. Return the *fraction*: 2 of the 4 tokens in \"а б\" is 0.5.")
    if abs(share - 50) < 1e-9:
        fail("You return 50, a percentage. Return the fraction, 0.5. The print line multiplies by 100 itself.")
    fail(f"In \"а б\" with \"а\" merged, the tokens are а, space and the two halves of б: 2 of 4 are partial, 0.5. You return {share:g}.")


@test("script_runs", "The script compares both tokenizers on the test set")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "pieces of a letter" not in stdout():
        fail("The comparison loop at the bottom is gone. Put it back to see both tokenizers.")
