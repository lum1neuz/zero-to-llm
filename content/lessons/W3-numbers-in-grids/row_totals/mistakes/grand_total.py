def row_totals(grid):
    total = 0
    for row in grid:
        total = total + sum(row)
    return total

counts = [[2, 1, 1],
          [0, 3, 1]]
print(row_totals(counts))
