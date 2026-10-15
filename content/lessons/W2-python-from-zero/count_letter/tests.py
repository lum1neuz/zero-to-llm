from harness import test, fail, learner_fn

# "aaaa" first: it tells "never adds 1" (0) apart from "returns inside the loop" (1)
CASES = [("aaaa", "a", 4), ("banana", "a", 3), ("emma", "m", 2), ("abc", "z", 0)]


def run(L, word, letter):
    count_letter = learner_fn(L, "count_letter")
    got = count_letter(word, letter)
    if got is Ellipsis:
        fail("`count` is still `...`, the placeholder. Start the counter at 0: nothing has been counted yet.")
    if got is None:
        fail("Your function gives back nothing. A function hands back its answer with `return`. Is `return count` still at the end?")
    return got


@test("returns_number", "count_letter gives back a whole number")
def _(L):
    got = run(L, "banana", "a")
    if not isinstance(got, int) or isinstance(got, bool):
        fail(f"It gave back {got!r}, which isn't a whole number. Return the counter itself: `return count`.")


@test("counts_matches", "It counts only the matching letters")
def _(L):
    for word, letter, want in CASES:
        got = run(L, word, letter)
        if got == want:
            continue
        if got == len(word) and want != len(word):
            fail(f"For \"{word}\" and \"{letter}\" you got {got}, which is every letter in the word. Only add 1 when the letter matches: `if ch == letter:`.")
        if got == 0:
            fail(f"For \"{word}\" and \"{letter}\" you got 0, so the counter never goes up. Inside the `if`, store the new value: `count = count + 1`. Writing just `count + 1` works out the sum but throws it away.")
        first = 1 if word[0] == letter else 0
        if got == first:
            fail(f"For \"{word}\" you got {got}. It looks like your function stops after the first letter. Is `return count` indented inside the loop? It should line up with `for`, so it runs after the loop is finished.")
        fail(f"For \"{word}\" and \"{letter}\" the answer is {want}, but you got {got}. Go through the word one letter at a time, on paper, and follow what your code does.")


@test("prints", "The script prints the count for banana")
def _(L):
    from harness import script_error, stdout
    err = script_error()
    if err:
        fail(f"Your script stopped: {err['diagnosis']}")
    if "3" not in stdout():
        fail("The last line should print 3, the number of a's in banana. Is `print(count_letter(\"banana\", \"a\"))` still there?")
