def to_percentages(row):
    total = sum(row)
    percents = []
    for value in row:
        # TODO: this value as a percent of the total
        percents.append(total / value * 100)
    return percents

print(to_percentages([5, 3, 2]))
