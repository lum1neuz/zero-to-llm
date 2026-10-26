---
id: "W3"
title: Numbers in grids
summary: Models keep their numbers in grids, like a spreadsheet. Learn rows and columns, add up a row, and turn counts into percentages.
phase: "W"
hours: 1.5
type: intro
status: ready
protected: false
cut: null
prerequisites: ["W2"]
papers: []
quality_metric:
  id: challenges_solved
  label: Challenges solved
  unit: count
systems_metric:
  id: minutes_spent
  label: Time spent on this lesson
  unit: min
quiz:
  - id: row_then_column
    q: In `grid[1][2]`, what do the two numbers mean?
    answer: 0
    options:
      - text: Row 1, then column 2
        why: "Yes. The first number picks a row (a list inside the grid). The second picks an item inside that row. Both count from 0."
      - text: Column 1, then row 2
        why: "It's the other way round. The grid is a list of rows, so the first number picks the row."
      - text: The 1st and 2nd numbers in the grid
        why: "They're positions, not values. Together they point at one cell: row 1, column 2."
  - id: why_percent
    q: Why turn counts into percentages?
    answer: 1
    options:
      - text: Percentages are smaller numbers, so they're faster
        why: "Speed isn't the reason. Percentages are about comparing fairly."
      - text: So rows with different totals can be compared, and each row shows how likely each option is
        why: "Right. 5 out of 10 and 50 out of 100 are both 50%. A row of percentages is exactly the \"how likely is each next letter\" table from W1."
      - text: Computers can't store whole numbers
        why: "Computers store whole numbers fine. Percentages make different rows comparable."
---

## Goal

Get comfortable with numbers arranged in a grid: rows across, columns down. You'll add up each row and turn a row of counts into percentages. This is exactly what the name-guessing model in lesson 0.3 does.

## Learn

```card
type: concept
id: grids_are_tables
eyebrow: Like a spreadsheet
title: A grid is rows and columns of numbers.
body: |
  You've seen this in any spreadsheet:

  | | col 0 | col 1 | col 2 |
  |---|---|---|---|
  | **row 0** | 2 | 1 | 1 |
  | **row 1** | 0 | 3 | 1 |

  In Python, a grid is a **list of rows**, and each row is a list:

  ```python
  grid = [[2, 1, 1],
          [0, 3, 1]]
  ```

  This grid has **2 rows** and **3 columns**.
```

```card
type: predict
id: pick_a_cell
eyebrow: Predict the output
title: What gets printed?
code: |
  grid = [[1, 2, 3],
          [4, 5, 6]]
  print(grid[1][0])
options:
  - text: "4"
    correct: true
    why: "`grid[1]` is row 1: `[4, 5, 6]`. Then `[0]` is the first item of that row: 4. Remember that counting starts at 0."
  - text: "2"
    why: "That would be `grid[0][1]`: row 0, item 1. The first number picks the row."
  - text: "1"
    why: "That's `grid[0][0]`. `grid[1]` is the second row, because counting starts at 0."
hint: Do it in two steps. What is grid[1]? Then take item 0 of that.
```

```card
type: fill_blank
id: find_six
eyebrow: Fill the blanks
title: Point at the 6.
code: |
  grid = [[1, 2, 3],
          [4, 5, 6]]
  print(grid[ [[0]] ][ [[1]] ])
tokens: ["0", "1", "2", "3"]
answer: ["1", "2"]
tips:
  "0": "Row 0 is `[1, 2, 3]`, and there's no 6 there. Which row has the 6?"
  "3": "Row 1 has three items, at positions 0, 1 and 2. There's no position 3."
why_ok: "6 is in row 1, at position 2: `grid[1][2]`."
why_wrong: "First find the row that contains 6, then count along that row starting from 0."
hint: The 6 is in the second row. Counting from 0, what number is the second row?
```

```card
type: calc
id: one_out_of_four
eyebrow: Do the math
title: "A row is [2, 1, 1]. What percent of its total is the first number?"
given:
  - { label: Row total, value: "2 + 1 + 1 = 4" }
  - { label: First number, value: "2" }
unit: "%"
answer: { min: 50, max: 50 }
placeholder: "0"
why_ok: "2 out of 4 is a half, which is 50%. The whole row becomes 50%, 25%, 25%, which adds up to 100%."
why_wrong: "Divide the number by the row's total, then multiply by 100: 2 ÷ 4 × 100."
hint: "Percent = (part ÷ total) × 100."
```

```card
type: reorder
id: percent_steps
eyebrow: Put it in order
title: Put the steps for turning a row into percentages in order.
lines:
  - total = sum(row)
  - "for value in row:"
  - "    percent = value / total * 100"
  - "    print(percent)"
shuffle: [2, 0, 3, 1]
why_ok: "First the total, because every percent needs it. Then, for each value, work out its percent and show it."
why_wrong: "You can't work out a percent before you know the total, and the pushed-in lines belong inside the loop."
hint: Which number does every percent depend on? Work that out first.
```

```card
type: concept
id: tensor_preview
eyebrow: A first look at tensors
title: From lesson 0.0 on, grids are kept in tensors.
body: |
  A **tensor** is a grid that the computer can do math on very fast, all at once. You make one from a list of rows:

  ```python
  t = torch.tensor([[2, 1, 1],
                    [0, 3, 1]])
  print(t.shape)
  ```

  This prints `torch.Size([2, 3])`: **2 rows, 3 columns**. That's the tensor's **shape**.

  The row totals you'll compute with a loop in this lesson come out of a tensor in one step: `t.sum(dim=1)` gives `[4, 4]`. You don't need this yet. Lesson 0.0 explains it slowly.
```

```card
type: quiz
ref: row_then_column
```

```card
type: quiz
ref: why_percent
```

## Build steps

### Add up each row

```challenge
id: row_totals
file: grids.py
minutes: 15
start: { code: "" }
setup:
  - say: "Here's a grid of counts: two rows of three numbers. We want each row's total in a list: 2 + 1 + 1 is 4, and 0 + 3 + 1 is 4, so the answer is [4, 4]."
    pause: 3.5
  - say: "The loop already visits each row. Your turn: add this row's total to the list. sum() adds up a list for you."
    type_file: row_totals/stub.py
    pause: 1
tests: row_totals/tests.py
solution: row_totals/solution.py
alternatives:
  - row_totals/alternatives/inner_loop.py
hints:
  start:
    try: |
      print(counts[0])
      print(sum(counts[0]))
    see: |
      [2, 1, 1]
      4
    next: "That's the first row of the grid, and its total. Inside the loop, `row` is one of these rows. Put its total on the end of `totals`, and the loop does the rest."
  logic: |
    totals starts as an empty list
    for each row in grid:
      add up the numbers in this one row
        → put that total on the end of totals
    after the loop
      → return totals   (one total per row)
  tools: |
    add up the numbers in one row
      sum(seq)         adds every number in a list
    put the total on the end of totals
      lst.append(x)    adds x to the end of the list lst
  nudge: "Inside the loop, `row` is one list of numbers. What do you want to put into `totals` for it?"
  concept: "`sum(row)` adds up one row. `totals.append(x)` adds `x` to the end of the `totals` list. Combine the two."
  example: |
    Trace it on `grid = [[1, 2], [3, 4], [5, 6]]`, with `totals` starting as `[]`:

    - First `row` is `[1, 2]`. Its total is `3`, so `totals` becomes `[3]`.
    - Second `row` is `[3, 4]`. Its total is `7`, so `totals` becomes `[3, 7]`.
    - Third `row` is `[5, 6]`. Its total is `11`, so `totals` becomes `[3, 7, 11]`.

    Three rows in, three totals out. If you got `[9, 12]` instead, you added down the columns.
  skeleton: |
    totals.______(___(row))
  solution: |
    totals.append(sum(row))
on_solve: "Solved. One total per row: in lesson 0.3 these become the totals that turn letter counts into chances."
mistakes:
  - { file: row_totals/mistakes/placeholder.py, fails: one_per_row, says: "comes back empty" }
  - { file: row_totals/mistakes/grand_total.py, fails: one_per_row, says: "everything added together" }
  - { file: row_totals/mistakes/appends_row.py, fails: one_per_row, says: "the rows themselves" }
  - { file: row_totals/mistakes/appends_length.py, fails: one_per_row, says: "how many numbers each row has" }
  - { file: row_totals/mistakes/column_totals.py, fails: one_per_row, says: "column totals" }
```

### Turn counts into percentages

```challenge
id: to_percentages
file: percents.py
minutes: 15
start: { code: "" }
setup:
  - say: "Now turn a row of counts into percentages. In [5, 3, 2] the total is 10, so 5 is 50%, 3 is 30% and 2 is 20%."
    pause: 3.5
  - say: "The total is worked out for you. Your turn: append each value as a percent of the total."
    type_file: to_percentages/stub.py
    pause: 1
tests: to_percentages/tests.py
solution: to_percentages/solution.py
alternatives:
  - to_percentages/alternatives/times_first.py
hints:
  start:
    try: |
      row = [5, 3, 2]
      total = sum(row)
      print(total)
      print(row[0] / total)
    see: |
      10
      0.5
    next: "That's the total, and the first value as a fraction of it: 5 out of 10 is `0.5`. A percent is that fraction times 100. Work out the percent for the first value by hand, then ask which loop variable stands for it."
  logic: |
    total = the whole row added up   (already done for you)
    percents starts as an empty list
    for each value in the row:
      fraction = value ÷ total
        → percent = fraction × 100
        → put it on the end of percents
    after the loop
      → return percents
  tools: |
    fraction = value ÷ total, then × 100
      plain arithmetic: a / b   and   a * b
    put the percent on the end of percents
      lst.append(x)    adds x to the end of the list lst
  nudge: What's 5 as a percent of 10? Write down how you worked it out.
  concept: "A percent is the value divided by the total, times 100. Append that to `percents` for each value."
  example: |
    Trace it on `row = [1, 3]`. The total is `4`:

    - `value = 1`: `1 / 4` is `0.25`, times 100 is `25.0`. `percents` becomes `[25.0]`.
    - `value = 3`: `3 / 4` is `0.75`, times 100 is `75.0`. `percents` becomes `[25.0, 75.0]`.

    Check: `25.0 + 75.0` is `100.0`. Any row of percents adds up to 100, because every value is a part of the same total.
  skeleton: |
    percents.append(_____ / _____ * ___)
  solution: |
    percents.append(value / total * 100)
on_solve: "Solved. A row of percentages that adds up to 100 is a row of probabilities. That's the core of the name model in lesson 0.3."
mistakes:
  - { file: to_percentages/mistakes/placeholder.py, fails: percents, says: "comes back empty" }
  - { file: to_percentages/mistakes/fractions.py, fails: percents, says: "fraction times 100" }
  - { file: to_percentages/mistakes/upside_down.py, fails: percents, says: "upside down" }
  - { file: to_percentages/mistakes/over_100.py, fails: percents, says: "divided each value by 100" }
```

## Hints

- Draw the grid on paper with row and column numbers starting at 0. It makes `grid[r][c]` much easier.
- If a test says you got column totals, you added downwards. A row goes across.

## Done when

- [ ] Finish the Learn cards {#cards auto=learn}
- [ ] Add up each row {#rows auto=challenge:row_totals}
- [ ] Turn counts into percentages {#percents auto=challenge:to_percentages}
