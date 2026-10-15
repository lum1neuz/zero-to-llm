from harness import test, fail, learner_fn

CASES = [([5, 3, 2], [50.0, 30.0, 20.0]), ([1, 3], [25.0, 75.0]), ([4], [100.0])]


def close(a, b):
    return isinstance(a, list) and len(a) == len(b) and all(isinstance(x, (int, float)) and abs(x - y) < 1e-6 for x, y in zip(a, b))


@test("percents", "Each value becomes a percent of the row's total")
def _(L):
    to_percentages = learner_fn(L, "to_percentages")
    for row, want in CASES:
        got = to_percentages(list(row))
        if close(got, want):
            continue
        if got is None:
            fail("Your function gives back nothing. Is `return percents` still at the end?")
        if got == []:
            fail("`percents` comes back empty. Inside the loop, append each value's percent with `percents.append(...)`.")
        total = sum(row)
        if close(got, [v / total for v in row]):
            fail(f"For {row} you got {got}. Those are fractions of the total (0.5 is a half). A percent is a fraction times 100.")
        if close(got, [total / v for v in row]) or close(got, [total / v * 100 for v in row]):
            fail(f"For {row} you got {got}, which is the total divided by each value: upside down. Divide the value by the total.")
        if close(got, [v / 100 for v in row]):
            fail(f"For {row} you got {got}. You divided each value by 100, but a percent compares the value to the row's total ({total}), not to 100.")
        fail(f"For {row} the percents are {want}, but you got {got}.")


@test("adds_to_100", "The percents in a row add up to 100")
def _(L):
    to_percentages = learner_fn(L, "to_percentages")
    got = to_percentages([7, 2, 1])
    if not isinstance(got, list) or not got:
        fail("Fix the test above first.")
    s = sum(got)
    if abs(s - 100) > 1e-6:
        fail(f"For [7, 2, 1] your percents add up to {s:g}. A whole row is always 100%, because every value is part of the same total.")
