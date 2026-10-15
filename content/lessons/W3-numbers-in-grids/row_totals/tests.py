from harness import test, fail, learner_fn

# Not square, so row totals and column totals have different lengths.
CASES = [([[1, 2], [3, 4], [5, 6]], [3, 7, 11]), ([[2, 1, 1], [0, 3, 1]], [4, 4]), ([[5]], [5])]


@test("one_per_row", "It gives back one total for each row")
def _(L):
    row_totals = learner_fn(L, "row_totals")
    for grid, want in CASES:
        got = row_totals([list(r) for r in grid])
        if got == want:
            continue
        if got is None:
            fail("Your function gives back nothing. Is `return totals` still at the end?")
        if isinstance(got, (int, float)):
            fail(f"For {grid} you got the single number {got}: everything added together. We want one total per row, in a list, like {want}.")
        if got == []:
            fail("`totals` comes back empty. Inside the loop, add each row's total to it with `totals.append(...)`.")
        if got == grid:
            fail(f"You got the rows themselves back, {got}. Add up each row before appending it. `sum(row)` adds up one row.")
        if got == [len(r) for r in grid]:
            fail(f"You got {got}: how many numbers each row has. We want what they add up to. `sum(row)`, not `len(row)`.")
        columns = [sum(r[c] for r in grid) for c in range(len(grid[0]))]
        if got == columns:
            fail(f"For {grid} you got {got}. Those are the column totals, added downwards. We want each row added across: {want}.")
        fail(f"For {grid} the row totals are {want}, but you got {got}.")
