from harness import test, fail, warn, learner_fn, LearnerError

SMALL = "aaabdaaabac"


def ref_stats(ids):
    counts = {}
    for p in zip(ids, ids[1:]):
        counts[p] = counts.get(p, 0) + 1
    return counts


def ref_merge(ids, pair, idx):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def check_step1(L):
    """Warn (don't fail) when step 1's functions have drifted since they were solved."""
    try:
        ok = L["get_stats"]([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1} and L["merge"]([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: get_stats or merge no longer passes step 1's tests")


def run_train(L, text, vocab_size):
    fn = learner_fn(L, "train")
    try:
        out = fn(text, vocab_size)
    except LearnerError as e:
        if "range(0, 256)" in str(e):
            fail("`bytes(...)` only takes numbers up to 255, but merged tokens have ids from 256 up. "
                 "A new token stands for the bytes its two parts already stand for: look them up in `vocab` and join them.")
        raise
    if not isinstance(out, tuple) or len(out) != 2:
        fail("`train` should return two things, `merges, vocab`. Keep the return line as it was.")
    merges, vocab = out
    if any(p is Ellipsis for p in merges) or any(v is Ellipsis for v in merges.values()):
        fail("`pair` or `idx` is still `...`, the placeholder.")
    return merges, vocab


def show(b):
    return repr(bytes(b)) if isinstance(b, (bytes, bytearray)) else repr(b)


@test("first_merge", "The first merge is the most frequent pair")
def _(L):
    check_step1(L)
    merges, _vocab = run_train(L, SMALL, 257)
    if len(merges) != 1:
        fail(f"`train(text, 257)` should make exactly 1 merge (257 − 256), but made {len(merges)}.")
    pair = next(iter(merges))
    if pair == (97, 97):
        return
    stats = ref_stats(list(SMALL.encode()))
    seen = stats.get(pair, 0)
    if pair == max(stats):
        fail(f"Your first merge is {pair}, the 'largest' pair. `max(stats)` compares the pairs themselves, not their counts. "
             f"(97, 97) 'aa' appears 4 times. Ask `max` to compare by count.")
    if seen == min(stats.values()):
        fail(f"Your first merge is {pair}, which appears only {seen} time in 'aaabdaaabac'. (97, 97) 'aa' appears 4 times. Which pair should win?")
    fail(f"Your first merge is {pair}. On 'aaabdaaabac' the most frequent pair is (97, 97), 'aa', seen 4 times.")


@test("new_ids", "New tokens get ids 256, 257, 258, …")
def _(L):
    merges, _vocab = run_train(L, SMALL, 259)
    ids = list(merges.values())
    if ids == [256, 257, 258]:
        return
    if len(ids) > 1 and len(set(ids)) == 1:
        fail(f"Every merge got id {ids[0]}, so later merges overwrite earlier ones. Each new token needs its own id, counting up with `i`.")
    if ids[:1] == [0]:
        fail("Your first new token has id 0, but ids 0–255 are already the 256 bytes. New tokens start at 256.")
    if ids[:1] == [255]:
        fail("Your first new token has id 255, which is already the byte 0xFF. New tokens start at 256.")
    fail(f"Your three merges got ids {ids}. They should be 256, 257 and 258.")


@test("vocab_bytes", "Each new token stands for its two parts' bytes, joined")
def _(L):
    merges, vocab = run_train(L, SMALL, 259)
    for (a, b), idx in merges.items():
        got = vocab.get(idx)
        if got is Ellipsis:
            fail("`vocab[idx]` is still `...`, the placeholder.")
        if not isinstance(got, (bytes, bytearray)):
            fail(f"`vocab[{idx}]` is {got!r}, a {type(got).__name__}. Every vocab entry should be bytes, so decode can join them.")
        want = vocab[a] + vocab[b] if isinstance(vocab.get(a), bytes) and isinstance(vocab.get(b), bytes) else None
        if got != want:
            if want is not None and got == vocab[a]:
                fail(f"Token {idx} merges {a} and {b}, so it stands for {show(want)}, but yours stands for {show(got)}, only the first part.")
            fail(f"Token {idx} merges {a} ({show(vocab.get(a))}) and {b} ({show(vocab.get(b))}), so it should stand for {show(want)}, not {show(got)}.")


@test("trains_story", "Every merge on the story is a most frequent pair at its turn")
def _(L):
    story = L.get("text")
    if not isinstance(story, str):
        fail("`text`, the story at the top, is gone. The tests train on it.")
    merges, vocab = run_train(L, story, 276)
    if len(merges) != 20:
        fail(f"`train(text, 276)` should make 276 − 256 = 20 merges, but made {len(merges)}.")
    ids = list(story.encode("utf-8"))
    for n, (pair, idx) in enumerate(merges.items()):
        stats = ref_stats(ids)
        best = max(stats.values())
        if stats.get(pair, 0) != best:
            fail(f"Merge {n + 1} picks {pair}, seen {stats.get(pair, 0)} times, but another pair is seen {best} times at that point.")
        ids = ref_merge(ids, pair, idx)
