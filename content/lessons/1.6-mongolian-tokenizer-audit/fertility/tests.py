from harness import test, fail, learner_fn, script_error, stdout

# "аб аб": а = D0 B0, б = D0 B1. With these merges each "аб" becomes one token,
# so the sentence is 3 tokens (аб, space, аб), 2 words, 5 characters and 9 bytes.
AB = {(208, 176): 256, (208, 177): 257, (256, 257): 258}


def ref_fertility(sentences, merges):
    tokens = 0
    for s in sentences:
        ids = list(s.encode("utf-8"))
        while len(ids) >= 2:
            pairs = list(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                    out.append(merges[pair])
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        tokens += len(ids)
    return tokens / sum(len(s.split()) for s in sentences)


def uses_script_merges(L, f, sentences):
    for name in ("mn_merges", "en_merges"):
        m = L.get(name)
        if isinstance(m, dict) and abs(f - ref_fertility(sentences, m)) < 1e-9 and abs(f - ref_fertility(sentences, AB)) > 1e-9:
            fail(f"Your fertility matches `{name}` from the script, not the merges passed in. "
                 "Use the function's own `merges` argument, so the same function can audit any tokenizer.")


def fert(L, sentences, merges):
    fn = learner_fn(L, "fertility")
    out = fn(list(sentences), merges)
    try:
        return float(out)
    except (TypeError, ValueError):
        fail(f"`fertility` should return a number, but it returns {type(out).__name__}.")


@test("tokens_per_word", "Fertility is tokens divided by words")
def _(L):
    f = fert(L, ["аб аб"], AB)
    if abs(f - 1.5) < 1e-9:
        return
    if abs(f - 2.5) < 1e-9:
        fail("For \"аб аб\" you get 2.5, which is characters per word. Count *tokens*: encode the sentence with the merges you're given.")
    if abs(f - 4.5) < 1e-9:
        fail("For \"аб аб\" you get 4.5, which is bytes per word: the merges never got applied. Encode with the `merges` argument.")
    uses_script_merges(L, f, ["аб аб"])
    if abs(f - 3.0) < 1e-9:
        fail("For \"аб аб\" you get 3.0: 3 tokens divided by 1. That's tokens per *sentence*. Count the words: split each sentence on spaces.")
    fail(f"For \"аб аб\" with merges that make each \"аб\" one token, you get {f:g}. It's 3 tokens for 2 words: 1.5.")


@test("all_sentences", "Counts every sentence, with the merges it's given")
def _(L):
    f = fert(L, ["аб аб", "аб"], AB)
    if abs(f - 4 / 3) < 1e-9:
        pass
    elif abs(f - 1.25) < 1e-9:
        fail("You averaged each sentence's fertility, (1.5 + 1) / 2 = 1.25. Fertility over a set is total tokens over total words, "
             "4 / 3 here, so long sentences count for more.")
    elif abs(f - 1.5) < 1e-9 or abs(f - 1.0) < 1e-9:
        fail("Only one of the two sentences got counted. Add up tokens and words over all of them.")
    else:
        fail(f"For [\"аб аб\", \"аб\"] you get {f:g}. That's 4 tokens over 3 words: 1.333.")
    if abs(fert(L, ["аб аб"], {}) - 4.5) > 1e-9:
        fail("With no merges, \"аб аб\" is 9 bytes, so 9 tokens over 2 words, 4.5. "
             "Your function ignores its `merges` argument and uses a tokenizer from the script.")


@test("script_runs", "The script compares the three tokenizers")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "Mongolian BPE" not in stdout():
        fail("The comparison loop at the bottom is gone. Put it back to see the three fertilities.")
