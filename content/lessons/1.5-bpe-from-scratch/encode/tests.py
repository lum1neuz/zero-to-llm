from harness import test, fail, warn, learner_fn, LearnerError, script_error, stdout

BYTES = {i: bytes([i]) for i in range(256)}
# Training learned "bc" first, then "ab". On "abcab", "ab" is the more frequent pair,
# so applying merges by frequency instead of by training order gives different tokens.
ORDER_MERGES = {(98, 99): 256, (97, 98): 257}
ORDER_VOCAB = {**BYTES, 256: b"bc", 257: b"ab"}


def ref_train(text, vocab_size):
    ids, merges, vocab = list(text.encode("utf-8")), {}, dict(BYTES)
    for i in range(vocab_size - 256):
        counts = {}
        for p in zip(ids, ids[1:]):
            counts[p] = counts.get(p, 0) + 1
        pair = max(counts, key=counts.get)
        idx = 256 + i
        out, j = [], 0
        while j < len(ids):
            if j < len(ids) - 1 and (ids[j], ids[j + 1]) == pair:
                out.append(idx)
                j += 2
            else:
                out.append(ids[j])
                j += 1
        ids = out
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab, ids


class tokenizer:
    """Run the learner's encode/decode against a given merges and vocab, then put theirs back."""
    def __init__(self, L, merges, vocab):
        self.g = getattr(L.get("encode") or L.get("decode"), "__globals__", {})
        self.new = {"merges": merges, "vocab": vocab}

    def __enter__(self):
        self.saved = {k: self.g.get(k) for k in self.new}
        self.g.update(self.new)

    def __exit__(self, *exc):
        self.g.update(self.saved)


def decode(L, ids):
    fn = learner_fn(L, "decode")
    try:
        return fn(list(ids))
    except LearnerError as e:
        msg = str(e)
        if "'list' object has no attribute 'decode'" in msg:
            fail("`data` is a list of byte strings. Join them into one bytes object first, then decode it once.")
        if "expected str instance, bytes found" in msg:
            fail("`\"\".join(...)` joins strings, but each vocab entry is bytes. Join bytes with a bytes separator: `b\"\".join(...)`.")
        if "range(0, 256)" in msg:
            fail("`bytes(ids)` only works for raw bytes, but merged tokens have ids from 256 up. Look up what each id stands for in `vocab`.")
        if "'str' object has no attribute 'decode'" in msg or "can't decode" in msg:
            fail("You decoded each token on its own. A token can hold half of a character (Cyrillic letters are 2 bytes, emoji 4), "
                 "so join all the bytes first and decode them once at the end.")
        raise


def encode(L, text):
    fn = learner_fn(L, "encode")
    try:
        out = fn(text)
    except LearnerError as e:
        if "KeyError" in str(e):
            fail("`merges[p]` raises a KeyError for a pair training never merged. Most pairs in a text were never merged, "
                 "so give them a rank that can never win: `merges.get(p, float(\"inf\"))`.")
        raise
    if not isinstance(out, list):
        fail(f"`encode` should return a list of ids, but it returns {type(out).__name__}.")
    return out


@test("decode_bytes", "decode turns ids back into text")
def _(L):
    merges, vocab, ids = ref_train("aaabdaaabac", 259)
    with tokenizer(L, merges, vocab):
        out = decode(L, ids)
    if out != "aaabdaaabac":
        fail(f"`decode({ids})` gives {out!r}. With this vocab it should give 'aaabdaaabac'.")
    # Token 256 ends with the first byte of "С" (0xD0 0xA1); the second byte is a token of its own.
    with tokenizer(L, {(97, 208): 256}, {**BYTES, 256: b"a\xd0"}):
        out = decode(L, [256, 161])
    if out != "aС":
        fail(f"`decode([256, 161])` gives {out!r}, but token 256 is b'a\\xd0' and 161 is b'\\xa1': together they spell 'aС'. "
             "A character can be split across tokens, so join all the bytes first and decode once.")


@test("encode_earliest", "encode applies merges in the order training learned them")
def _(L):
    with tokenizer(L, ORDER_MERGES, ORDER_VOCAB):
        got = encode(L, "abcab")
    if got == [97, 256, 257]:
        return
    if got == [257, 99, 257]:
        fail("\"abcab\" became [257, 99, 257]: you merged the most frequent pair, (97, 98) 'ab', first. "
             "But training learned (98, 99) 'bc' before 'ab'. Apply merges in training order, earliest first, "
             "or you make tokens training never saw.")
    if got == [97, 98, 99, 97, 98]:
        fail("\"abcab\" came back as raw bytes: nothing got merged. If `pair` is still `...`, `pair not in merges` ends the loop at once. "
             "Of the pairs in `stats`, which one did training merge first?")
    fail(f"With merges bc → 256 then ab → 257, \"abcab\" should encode to [97, 256, 257], but gives {got}.")


@test("round_trip", "decode(encode(s)) gives back every line of the test set")
def _(L):
    story = L.get("text")
    if not isinstance(story, str) or not isinstance(L.get("merges"), dict) or not isinstance(L.get("vocab"), dict):
        fail("`text`, `merges` or `vocab` is missing. The script above trains them; keep that part as it was.")
    for s in L.get("test_set") or ["Сайн байна уу?", "🙂🐢"]:
        ids = encode(L, s)
        unknown = [i for i in ids if i not in L["vocab"]]
        if unknown:
            fail(f"`encode({s!r})` makes id {unknown[0]}, which isn't in `vocab`.")
        back = decode(L, ids)
        if back != s:
            fail(f"{s!r} comes back as {back!r}.")
    ids = encode(L, story)
    if len(ids) >= len(story.encode("utf-8")):
        fail("Encoding the story gives as many tokens as bytes, so no merges were applied.")


@test("script_runs", "The script prints the round trip and the encoding speed")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "MB/s" not in stdout():
        fail("The timing lines at the bottom are gone. Put them back: they print your systems metric.")
