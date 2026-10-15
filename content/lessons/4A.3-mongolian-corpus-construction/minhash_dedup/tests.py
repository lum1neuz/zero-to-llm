from harness import test, fail, warn, learner_fn, as_float, script_error

SET = {"Монгол Улсын Их", "Улсын Их Хурлын", "Их Хурлын намрын", "Хурлын намрын ээлжит", "намрын ээлжит чуулган"}


def ref_h(seed, s):
    x = 2166136261
    for b in seed.to_bytes(4, "little") + s.encode("utf-8"):
        x = ((x ^ b) * 16777619) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    return x ^ (x >> 16)


def ref_sig(shingle_set, k):
    return [min(ref_h(seed, s) for s in shingle_set) for seed in range(k)]


@test("signature", "minhash gives k minimums, one per seed")
def _(L):
    fn = learner_fn(L, "minhash")
    out = fn(SET, 16)
    if out is Ellipsis:
        fail("`minhash` still returns `...`, the placeholder.")
    if out is None:
        fail("`minhash` returns nothing. It needs a `return`.")
    if isinstance(out, (set, frozenset)):
        fail("`minhash` returns a set, which loses the order. A signature is a list: position i holds seed i's minimum.")
    if not isinstance(out, (list, tuple)):
        fail(f"`minhash` should return a list of k numbers, but it returns {type(out).__name__}.")
    sig = list(out)
    good = ref_sig(SET, 16)
    if sig == good:
        if len(fn(SET)) != 64:
            fail("With no `k` given, the signature should have 64 entries. Keep the default `k=64`.")
        return
    if len(sig) == len(SET):
        fail(f"Your signature has {len(SET)} entries, one per shingle. It should have k = 16, one per seed: "
             "for each seed, the smallest hash over all the shingles.")
    if len(sig) != 16:
        fail(f"Your signature has {len(sig)} entries, but k = 16 seeds should give 16.")
    if len(set(sig)) == 1:
        fail("Every entry of your signature is the same number: they all use one seed. Position i should use seed i.")
    if sig == [max(ref_h(seed, s) for s in SET) for seed in range(16)]:
        warn("You take the largest hash per seed. That works just as well in theory, but MinHash libraries use the smallest, so stick with min to match them.")
        return
    fail("Your signature doesn't match: position i should be the smallest `h(i, s)` over every shingle s in the set.")


@test("estimate_value", "estimate is the share of positions that agree")
def _(L):
    out = learner_fn(L, "estimate")([1, 2, 3, 4, 5], [1, 9, 3, 8, 5])
    if out is Ellipsis:
        fail("`estimate` still returns `...`, the placeholder.")
    v = as_float(out)
    if v is None:
        fail(f"`estimate` should return a number, but it returns {type(out).__name__}.")
    if abs(v - 0.6) < 1e-9:
        return
    if abs(v - 3) < 1e-9:
        fail("You return how many positions agree, 3. The estimate is a share: divide by the signature length.")
    if abs(v - 3 / 7) < 1e-9:
        fail("You compare the signatures as sets of numbers. Position matters: entry i of each is seed i's minimum, "
             "and the chance those two match is the Jaccard. Compare position by position.")
    if abs(v - 0.4) < 1e-9:
        fail("That's the share of positions that *disagree*. Count the ones that agree.")
    fail(f"[1, 2, 3, 4, 5] and [1, 9, 3, 8, 5] agree in 3 of 5 positions, so the estimate is 0.6. Yours is {v:.4f}.")


@test("dedup_runs", "The mirrored weather report is dropped, and only that one")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    kept, final = L.get("kept"), L.get("final")
    if not isinstance(final, list) or not isinstance(kept, list):
        fail("`final` is missing. Put the dedup loop at the bottom back.")
    if len(kept) != 7:
        fail(f"Step 2 should leave 7 documents, but `kept` has {len(kept)}. Do steps 1 and 2 still pass their tests?")
    dropped = [d for d in kept if d not in final]
    if dropped != [kept[1]]:
        fail(f"The loop dropped {len(dropped)} document(s). It should drop exactly one, doc 1, the weather report mirrored with a source line. "
             "Do step 3's `shingles` and `jaccard` still pass their tests?")
