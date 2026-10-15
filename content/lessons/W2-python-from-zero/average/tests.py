from harness import test, fail, learner_fn

CASES = [([2, 4, 9], 5.0), ([10], 10.0), ([1, 2], 1.5), ([3, 3, 3, 3], 3.0)]


def run(L, numbers):
    average = learner_fn(L, "average")
    got = average(list(numbers))
    if got is Ellipsis:
        fail("Your function still returns `...`, the placeholder. Return the total divided by how many numbers there are.")
    if got is None:
        fail("Your function gives back nothing. Hand the answer back with `return`.")
    return got


@test("small_lists", "It averages a few small lists")
def _(L):
    for numbers, want in CASES:
        got = run(L, numbers)
        if isinstance(got, (int, float)) and abs(got - want) < 1e-9:
            continue
        total = sum(numbers)
        if got == total:
            fail(f"For {numbers} you got {got}, which is the total. An average is the total divided by how many numbers there are. How many are in the list? `len(numbers)` tells you.")
        if got == total // len(numbers) and total % len(numbers):
            fail(f"For {numbers} you got {got}, but the answer is {want}. `//` divides and then throws away everything after the decimal point. Use a single `/` to keep the .5.")
        if isinstance(got, (int, float)) and abs(got - total / 3) < 1e-9 and len(numbers) != 3:
            fail(f"For {numbers} you got {got:g}. You always divide by 3, which only works for lists of exactly 3 numbers. Use `len(numbers)` so it works for any length.")
        fail(f"For {numbers} the average is {want}, but you got {got}. Add the numbers up, then divide by how many there are.")


@test("prints", "The script prints the average of 2, 4 and 9")
def _(L):
    from harness import script_error, stdout
    err = script_error()
    if err:
        fail(f"Your script stopped: {err['diagnosis']}")
    if "5" not in stdout():
        fail("The last line should print 5.0. Is `print(average([2, 4, 9]))` still there?")
