from harness import test, fail, learner_fn, as_float, script_error

PHRASE = "Монгол Улсын Их Хурлын чуулган"
REPEAT = "цас орно цас орно цас орно"   # the same 3 words come round again


def shingle_set(L, text, n):
    out = learner_fn(L, "shingles")(text, n)
    if out is Ellipsis:
        fail("`shingles` still returns `...`, the placeholder.")
    if out is None:
        fail("`shingles` returns nothing. It needs a `return`.")
    return out


def want(text, n):
    words = text.split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def show(items):
    return "{" + ", ".join(f'"{s}"' for s in sorted(items)) + "}"


@test("shingle_set", "shingles gives every run of n words, as a set of strings")
def _(L):
    out = shingle_set(L, PHRASE, 3)
    if isinstance(out, (list, tuple)):
        rep = shingle_set(L, REPEAT, 3)
        if isinstance(rep, (list, tuple)) and len(rep) != len(set(rep)):
            fail(f"`shingles` returns a {type(out).__name__}, so a phrase that comes round twice is counted twice: \"{REPEAT}\" "
                 f"gives {len(rep)} shingles where {len(set(rep))} are distinct. Jaccard works on sets. Return a set.")
        fail(f"`shingles` returns a {type(out).__name__}. Jaccard works on sets, so return a set.")
    if not isinstance(out, (set, frozenset)):
        fail(f"`shingles` should return a set of strings, but it returns {type(out).__name__}.")
    if any(isinstance(s, tuple) for s in out):
        fail("Your shingles are tuples of words. Jaccard would work, but step 4's hash reads a string's bytes. "
             "Join each run of words into one string, with single spaces.")
    if not all(isinstance(s, str) for s in out):
        fail("Every shingle should be a string, the n words joined by single spaces.")
    good = want(PHRASE, 3)
    if out == good:
        out2 = shingle_set(L, PHRASE, 2)
        if out2 != want(PHRASE, 2):
            fail(f"With n = 2 you get {show(out2)}, but pairs of words would be {show(want(PHRASE, 2))}. Use the `n` you're given.")
        return
    if out == good - {"Их Хурлын чуулган"}:
        fail(f"The last run, \"Их Хурлын чуулган\", is missing. {len(PHRASE.split())} words hold {len(good)} runs of 3. "
             "Where does your `range` stop?")
    if all(len(s) == 3 for s in out):
        fail(f"Your shingles are runs of 3 *characters* ({show(list(out)[:3])}…). Here a shingle is 3 consecutive *words*.")
    fail(f"For \"{PHRASE}\" you get {show(out)}, but the runs of 3 words are {show(good)}.")


@test("jaccard_value", "jaccard is shared over all distinct shingles")
def _(L):
    a = {"w1", "w2", "w3", "w4"}
    b = {"w3", "w4", "w5"}
    out = learner_fn(L, "jaccard")(a, b)
    if out is Ellipsis:
        fail("`jaccard` still returns `...`, the placeholder.")
    v = as_float(out)
    if v is None:
        fail(f"`jaccard` should return a number, but it returns {type(out).__name__}.")
    if abs(v - 0.4) < 1e-9:
        return
    stories = {
        2.0: "You return the number of shared shingles. Jaccard is a share: shared over all the distinct shingles of both.",
        2 / 3: "You divide by the smaller set. That's the overlap coefficient, which calls a short page a duplicate of any long page that quotes it. Divide by the union.",
        2 / 7: "You divide by len(a) + len(b), which counts the shared shingles twice. Divide by the union, `a | b`.",
        2.5: "Upside down: union over shared. Jaccard is shared over union, so it runs from 0 to 1.",
        0.6: "That's 1 − Jaccard, the Jaccard distance. Return the similarity.",
    }
    for value, story in stories.items():
        if abs(v - value) < 1e-9:
            fail(story)
    fail(f"{show(a)} and {show(b)} share 2 shingles out of 5 distinct ones, so Jaccard is 0.4. Yours is {v:.4f}.")


@test("near_dup_pair", "The mirrored weather report scores above 0.7, and the extended Khural story below")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    kept = L.get("kept")
    if not isinstance(kept, list) or len(kept) != 7:
        fail("Step 2 should leave 7 documents in `kept`. Does `keep` still pass its tests?")
    sh, jac = L["shingles"], L["jaccard"]
    j01 = as_float(jac(sh(kept[0]), sh(kept[1])))
    j23 = as_float(jac(sh(kept[2]), sh(kept[3])))
    if j01 is None or not j01 > 0.7:
        fail(f"The weather report and its mirror (docs 0 and 1) score {j01}. They should be above 0.7. Does step 1's `normalize` still fix the Latin letters?")
    if j23 is None or not 0.4 < j23 < 0.7:
        fail(f"Docs 2 and 3 score {j23}. Doc 3 is doc 2 plus a sentence, so it should land between 0.4 and 0.7.")
