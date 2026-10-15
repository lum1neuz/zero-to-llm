def row_totals(grid):
    totals = []
    for row in grid:
        # TODO: add this row's total to the list
        t = 0
        for value in row:
            t = t + value
        totals.append(t)
    return totals

counts = [[2, 1, 1],
          [0, 3, 1]]
print(row_totals(counts))
