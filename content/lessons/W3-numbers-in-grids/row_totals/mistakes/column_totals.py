def row_totals(grid):
    totals = []
    for c in range(len(grid[0])):
        totals.append(sum(row[c] for row in grid))
    return totals

counts = [[2, 1, 1],
          [0, 3, 1]]
print(row_totals(counts))
