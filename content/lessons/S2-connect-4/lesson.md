---
id: "S2"
title: Connect 4
summary: Scale S1's search to Connect 4, with fast win checks, in-place rollouts, self-play training data and a depth-4 minimax opponent, then watch a self-play run learn to beat it.
phase: "S"
hours: 8
type: sidequest
status: ready
protected: false
cut: null
prerequisites:
  - S1
papers:
  - title: Mastering Chess and Shogi by Self-Play (AlphaZero)
    authors: Silver et al.
    year: 2017
    url: https://arxiv.org/abs/1712.01815
quality_metric:
  id: win_rate
  label: Win rate vs a depth-4 minimax bot
  unit: "%"
  note: 100 games against step 4's bot, 50 as X and 50 as O, played by your trained agent.
  inputs:
    - { id: win_rate, label: Win rate vs depth-4 minimax (%), hint: "Wins out of 100, from your run's last eval" }
  compute: value
  reference: 88
  pass: { min: 60 }
systems_metric:
  id: mcts_sims_per_s
  label: MCTS simulations per second
  unit: sims/s
  note: Step 2's speed test in the browser, 300 simulations from the empty board with your in-place rollout. rollout_copy manages about a sixth of that.
  inputs:
    - { id: sims_per_s, label: MCTS simulations per second, hint: "The rollout line of step 2's speed test" }
  compute: value
  reference: 8000
  pass: { min: 3000 }
quiz:
  - id: last_move_only
    q: Why can is_win look only at lines through the piece that was just dropped?
    answer: 0
    options:
      - text: Any four that wasn't there before has to include the new piece
        why: The board was checked after every earlier move, and only one cell has changed since. A new four must pass through that cell, so checking its four directions is exact.
      - text: Fours elsewhere on the board can't happen in Connect 4
        why: They can, but they'd have been found the move they were made, and the game would have ended there.
      - text: It's a shortcut that misses a few rare wins
        why: It misses none. The random-game test in step 1 compares it with a full check after every move of 200 games.
  - id: z_for_o
    q: X wins a self-play game. One stored position had O to move. Which z does that example get?
    answer: 1
    options:
      - text: "+1, because the game was won"
        why: "+1 is X's result. The value head learns from the mover's side, and for O this game was a loss."
      - text: "−1"
        why: z is the final result from the view of the player to move in that position. O went on to lose, so −1, and the net learns this position was bad for the side to move.
      - text: "0, because only the last position decided the game"
        why: Every position in the game gets the final result. That's noisy for early moves, but over thousands of games the value head averages it out.
  - id: explore_moves
    q: self_play_game samples its first 8 moves from the visit counts, then always plays the most visited move. Why sample early on?
    answer: 0
    options:
      - text: So games open differently and the data covers more positions
        why: Always playing the top move makes many games start the same way, so the net would see the same few openings over and over. Sampling spreads the games out; later moves still play the search's best.
      - text: Sampled moves are stronger than the most visited one
        why: The most visited move is the search's best guess. Sampling plays weaker moves on purpose, which is only worth it early, for variety.
      - text: Sampling makes each simulation faster
        why: Sampling happens once per move, after the search. It doesn't change how fast the simulations run.
---

## Goal

Take S1's search to a game it can't solve by brute force. Connect 4 has about 4.5 trillion positions, so speed and good training data start to matter. Write a **win check** that looks only around the last piece, **rollouts that play and undo** on one shared board, the code that turns **self-play games into (state, π, z) examples**, and a **depth-4 minimax bot** to measure against. Then watch a self-play training run learn to beat that bot.

## Learn

```card
type: concept
id: bigger_game
eyebrow: Same search, bigger game
title: 5,478 positions become 4.5 trillion.
body: |
  Tic-tac-toe's tree fits in memory; S1's perfect player looked at all of it. Connect 4 has 7 moves a turn and games up to 42 moves long, so no one solves it on the fly. MCTS still works, but each random rollout is now around 20 moves of noise, so it needs far more simulations to see anything. That makes two things matter: how many simulations a second you can run, and a value net that replaces the rollout's guess with a learned one.
```

```card
type: concept
id: play_and_undo
eyebrow: Speed
title: Don't copy the board. Play, then take it back.
visual:
  kind: bars
  max: 9000
  bars:
    - { label: copy + scan, value: 1400, text: "1,400/s" }
    - { label: play + undo, value: 8200, text: "8,200/s" }
body: |
  A simulation plays dozens of moves. Copying the 6×7 board for each one, then checking all 42 cells for a four after every move, wastes most of the time. Playing on one shared board and taking each move back afterwards costs nothing to copy, and a four can only pass through the piece just dropped, so that's the only place to look. The bars show MCTS simulations a second in the browser for both ways.
```

```card
type: predict
id: negative_index
eyebrow: Predict the output
title: Walking left from column 0. What prints?
code: |
  row = [1, 0, 0, 0, 0, 1, 1]    # bottom row: X in columns 0, 5 and 6
  c, count = 0 - 1, 0            # start one step left of column 0
  while c < 7 and row[c] == 1:
      count += 1
      c -= 1
  print(count)
options:
  - text: "0"
    why: That's the right answer for the game, but nothing in the loop stops c at −1.
  - text: "2"
    correct: true
    why: "In Python row[-1] is the last column and row[-2] the one before it. Both are X, so the walk wraps round the edge and counts columns 6 and 5. That's why run_length needs 0 <= c as well as c < COLS."
  - text: IndexError
    why: Negative indices don't raise in Python. They count from the end, so row[-1] through row[-7] are all valid.
hint: What does a negative index mean for a Python list?
```

```card
type: calc
id: count_windows
eyebrow: Do the math
title: How many lines of four cells fit on a 6×7 board?
given:
  - { label: Across, value: "4 starting columns in each of 6 rows" }
  - { label: Down, value: "3 starting rows in each of 7 columns" }
  - { label: Each diagonal, value: "3 starting rows × 4 starting columns" }
unit: windows
answer: { min: 69, max: 69 }
placeholder: "0"
why_ok: "24 across + 21 down + 12 + 12 on the diagonals = 69. The minimax bot's heuristic scores every one of them for every position it looks at."
why_wrong: "Add the four groups: 6 × 4 across, 3 × 7 down, and 3 × 4 for each of the two diagonals."
hint: A line of four across can start in columns 0 to 3, so 4 places in each row.
```

```card
type: fill_blank
id: label_z
eyebrow: Fill the blank
title: Label each position with how the game turned out.
code: |
  def to_examples(history, result):     # result: 1 X won, -1 O won, 0 draw
      return [(state, pi, [[0]]) for state, player, pi in history]
tokens: ["result * player", result, "-result * player", player]
answer: ["result * player"]
tips:
  result: That gives every position X's result. O's positions in a game X won would learn that they win.
  "-result * player": That flips every label, so the winner's positions would learn that they lose.
  player: That's who moved, not how the game ended.
why_ok: "+1 where the player to move went on to win, −1 where they lost, 0 everywhere in a draw. That's what the value head learns to predict."
why_wrong: "z is the final result seen from the side of the player to move in that state."
hint: X is 1 and O is −1. Multiplying by the player turns a result into that player's view of it.
```

```card
type: quiz
ref: last_move_only
```

```card
type: quiz
ref: z_for_o
```

```card
type: quiz
ref: explore_moves
```

## Build steps

### Win check around the last piece

```challenge
id: win_detection
file: connect4.py
minutes: 45
start: { code: "import math\nimport random\nimport time\n\n" }
setup:
  - say: "Connect 4 is 6 rows by 7 columns. Row 0 is the top, so a dropped piece lands in the highest-numbered row that's still empty. show prints the board."
    type_file: win_detection/board.py
    pause: 3.2
  - say: "A new four always runs through the piece that was just dropped, so there's no need to check the whole board. From the new piece, count matching pieces both ways along each of four directions. Your turn."
    type_file: win_detection/stub.py
    pause: 1.2
tests: win_detection/tests.py
solution: win_detection/solution.py
alternatives:
  - win_detection/alternatives/three_steps.py
  - win_detection/alternatives/any_direction.py
hints:
  start:
    try: |
      print(row, board[row][2])
      print(board[row][3] == board[row][2])
    see: |
      5 1
      True
    next: "That's the new piece at row 5 holding player 1, and the cell one step to its right holding the same player. `run_length` keeps taking that step (`row + dr`, `col + dc`) and counts while the cell is on the board and matches. Write it for one direction first and try it on this board."
  logic: |
    run_length(row, col, dr, dc):
      player = who sits at (row, col)
      count = 0, then step once away along (dr, dc)
      while this cell is on the board AND holds player:
        count = count + 1 → step once more
      (stops at an edge or at any other cell)
      return count   (the piece itself is not counted)

    is_win(row, col):
      for each direction: across, down, two diagonals:
        1 + run one way + run the opposite way is 4 or more?
          yes → return True
      none reached 4 → return False
  tools: |
    read who sits at the piece
      board[r][c]          the cell at row r, column c
    walk away while it matches
      while cond:          repeat as long as cond holds
      0 <= x < n           x is inside 0 ... n-1
      ROWS, COLS           the board's size
    count and step
      plain arithmetic: + and +=
    the opposite direction
      plain arithmetic: - flips a number's sign
    each of the four directions
      for a, b in pairs    unpacks two numbers per direction
      return True / False  leave early with an answer
  nudge: Any new four has to run through the piece that was just dropped. From that piece, which ways can a line go, and where does each walk have to stop?
  concept: "run_length walks from (row, col) one step of (dr, dc) at a time, counting cells that hold the same player, and stops at the first cell that's off the board or isn't theirs. Check both edges: `board[r][-1]` quietly reads the last column instead of failing. is_win adds the runs on both sides of the new piece, plus 1 for the piece itself, for each of the four directions, and wins at 4 or more."
  example: |
    Trace the starter board. Row 5 is the bottom row; X holds columns 0, 1 and 3, O sits on top of each, and X just dropped into column 2, so the piece is at `(5, 2)`:

    - Direction `(0, 1)`, to the right: `(5, 3)` is X, so the count is 1. `(5, 4)` is empty, stop. `run_length` gives `1`.
    - Opposite direction `(0, -1)`, to the left: `(5, 1)` and `(5, 0)` are X, then column −1 is off the board, stop. `run_length` gives `2`.
    - Total: 2 + 1 + 1 for the new piece = `4`, so `is_win` returns `True`.

    The other directions give less: going up, `(4, 2)` is empty, so the run is 0 and the total is just 1.
  skeleton: |
    # run_length
    player = board[row][col]
    count = 0
    r, c = row + ____, col + ____
    while 0 <= r < ____ and 0 <= c < ____ and board[r][c] == ______:
        count += 1
        r, c = r + dr, c + dc
    # is_win
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if ___ + run_length(board, row, col, dr, dc) + run_length(board, row, col, ____, ____) >= ___:
            return True
    return False
  solution: |
    # run_length
    player = board[row][col]
    count = 0
    r, c = row + dr, col + dc
    while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == player:
        count += 1
        r, c = r + dr, c + dc
    # is_win
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if 1 + run_length(board, row, col, dr, dc) + run_length(board, row, col, -dr, -dc) >= 4:
            return True
    return False
on_solve: Solved, and is_win agrees with a full-board check after every move of 200 random games. It looks at no more than 24 cells around the new piece. Step 2 makes the rollouts fast.
mistakes:
  - { file: win_detection/mistakes/placeholders.py, fails: run_length_counts, says: "`count` is still `...`" }
  - { file: win_detection/mistakes/counts_itself.py, fails: run_length_counts, says: "include (row, col) itself" }
  - { file: win_detection/mistakes/any_piece.py, fails: run_length_counts, says: "counted O's pieces as well" }
  - { file: win_detection/mistakes/no_stop.py, fails: run_length_counts, says: "kept going past the O" }
  - { file: win_detection/mistakes/wraps_round.py, fails: stays_on_board, says: "wrapped round" }
  - { file: win_detection/mistakes/walks_off.py, fails: stays_on_board, says: "walked off the board" }
  - { file: win_detection/mistakes/is_win_placeholder.py, fails: win_through_middle, says: "`is_win` still returns `...`" }
  - { file: win_detection/mistakes/one_way.py, fails: win_through_middle, says: "one direction only" }
  - { file: win_detection/mistakes/no_new_piece.py, fails: win_through_middle, says: "count the new piece itself" }
  - { file: win_detection/mistakes/no_rising_diagonal.py, fails: all_directions, says: "rising to the right" }
  - { file: win_detection/mistakes/no_vertical.py, fails: all_directions, says: "straight down column 0" }
  - { file: win_detection/mistakes/three_wins.py, fails: exactly_four, says: "is only three" }
  - { file: win_detection/mistakes/equals_four.py, fails: exactly_four, says: "five in a row contains a four" }
```

### Fast rollouts: play, then undo

```challenge
id: fast_rollouts
file: connect4.py
minutes: 50
start: { from: win_detection }
setup:
  - say: "Here's S1's search, adapted to Connect 4. It plays every move on one shared board and takes each one back out afterwards, so it never copies the board."
    type_file: fast_rollouts/search.py
    pause: 3.0
  - say: And here's the slow way to roll out a game. Copy the board, then check every cell for a four after each move.
    type_file: fast_rollouts/slow.py
    pause: 2.4
  - say: "Your turn: undo, and a rollout that plays on the board itself. The speed test at the bottom races the two."
    type_file: fast_rollouts/stub.py
    pause: 1.2
tests: fast_rollouts/tests.py
solution: fast_rollouts/solution.py
alternatives:
  - fast_rollouts/alternatives/break_on_win.py
  - fast_rollouts/alternatives/undo_any_order.py
  - fast_rollouts/alternatives/copy_rows.py
hints:
  start:
    try: |
      b = new_board()
      row = drop(b, 3, 1)
      print(row, is_win(b, row, 3))
      print([b[r][3] for r in range(ROWS)])
    see: |
      5 False
      [0, 0, 0, 0, 0, 1]
    next: "That's one move played right on the board: the piece landed in row 5, made no four, and column 3 now holds it at the bottom. A rollout repeats this with random open columns and must put every piece back afterwards. Write `undo` first: the top piece is the first cell from the top of the column that isn't 0."
  logic: |
    undo(col):
      look down column col, from the top row
      first cell that isn't empty?
        → set it back to empty → stop

    rollout:
      played = empty list, p = player, result = 0
      while nobody has won AND some column is open:
        pick a random open column → drop p's piece there
        add the column to played
        that piece made four?
          yes → result = 1 if the starting player made it, else -1
        hand the turn to the other side
      undo each column in played, newest first
      → hand back result   (1 win, -1 loss, 0 draw)
  tools: |
    undo: find the top piece
      range(n)             0, 1, ... n-1
      board[r][c]          read or set one cell
      return               leave the function early
    pick and drop a random move
      legal_moves(board)   the open columns
      rng.choice(lst)      a random item of lst
      drop(board, c, p)    puts p's piece in c, returns its row
      lst.append(x)        adds x at the end
    did it win?
      is_win(board, r, c)  did the piece at r, c make four?
      a if cond else b     picks a or b
    take everything back
      reversed(lst)        lst from last item to first
      undo(board, c)       takes the top piece out of c
  nudge: drop puts a piece in the lowest empty cell of a column. Which piece should undo take out? For rollout, keep a list of every column you play, so you can take them all back at the end.
  concept: "undo scans column col from the top (row 0) down and clears the first piece it meets. rollout drops random legal moves, remembers each column in `played`, checks is_win only at the new piece, and stops at a win or a full board. Then it undoes `played` newest first and returns the result for the player who started: 1 if they made the winning move, −1 if the other side did, 0 for a draw."
  example: |
    Trace a short rollout from an empty board with `player = 1` (X). Say the random picks are columns `0, 1, 0, 1, 0, 1, 0`:

    - X drops into column 0 and lands in row 5. O goes in column 1, row 5. X column 0, row 4. O column 1, row 4. X column 0, row 3. O column 1, row 3.
    - X's 4th piece in column 0 lands in row 2, and `is_win` finds four in a column. `p` is X, the player who started, so `result = 1`.
    - `played` is `[0, 1, 0, 1, 0, 1, 0]`. Undo it in reverse: column 0, then 1, then 0, and so on. Each `undo` clears the topmost piece, so the board ends up empty.

    If it were O who made the four, the same rollout would return `-1`. A rollout that fills the board with no four returns `0`.
  skeleton: |
    # undo
    for row in range(______):
        if board[row][col] != 0:
            board[row][col] = ___
            return
    # rollout
    played = []
    p = player
    result = 0
    while result == 0 and ________(board):
        col = rng.choice(legal_moves(board))
        row = drop(board, col, ___)
        played.append(___)
        if is_win(board, row, col):
            result = 1 if ________ else -1
        p = -p
    for col in ________(played):
        undo(board, col)
    return result
  solution: |
    # undo
    for row in range(ROWS):
        if board[row][col] != 0:
            board[row][col] = 0
            return
    # rollout
    played = []
    p = player
    result = 0
    while result == 0 and legal_moves(board):
        col = rng.choice(legal_moves(board))
        row = drop(board, col, p)
        played.append(col)
        if is_win(board, row, col):
            result = 1 if p == player else -1
        p = -p
    for col in reversed(played):
        undo(board, col)
    return result
on_solve: "Solved. Compare the two lines: playing on one board and checking only around the last piece runs about 6× more simulations a second than copying and checking everything. More simulations in the same time means stronger play. Note your rollout line: it's this lesson's systems metric."
mistakes:
  - { file: fast_rollouts/mistakes/undo_placeholder.py, fails: undo_top_piece, says: "still the `...` placeholder" }
  - { file: fast_rollouts/mistakes/undo_bottom.py, fails: undo_top_piece, says: "the first one dropped" }
  - { file: fast_rollouts/mistakes/undo_whole_column.py, fails: undo_top_piece, says: "emptied all of column 2" }
  - { file: fast_rollouts/mistakes/rollout_placeholder.py, fails: board_unchanged, says: "`rollout` returns nothing" }
  - { file: fast_rollouts/mistakes/no_undo.py, fails: board_unchanged, says: "Undo each move you made" }
  - { file: fast_rollouts/mistakes/undo_last_only.py, fails: board_unchanged, says: "new pieces on the board" }
  - { file: fast_rollouts/mistakes/who_won.py, fails: result_for_player, says: "That's *who* won" }
  - { file: fast_rollouts/mistakes/no_draw_stop.py, fails: result_for_player, says: "filled up with nobody winning" }
  - { file: fast_rollouts/mistakes/no_speed_test.py, fails: script_runs, says: "didn't print both lines" }
```

### Self-play: games into training data

```challenge
id: self_play_data
file: connect4.py
minutes: 35
start: { from: fast_rollouts }
setup:
  - say: visit_policy comes over from S1, with 7 columns instead of 9 cells.
    type_file: self_play_data/policy.py
    pause: 2.0
  - say: "In self-play the search plays both sides. Each move stores the position, who was to move, and the search's visit policy. When the game ends, every stored position gets labelled with how the game turned out for the player who was to move there. Write snapshot and to_examples."
    type_file: self_play_data/stub.py
    pause: 1.2
tests: self_play_data/tests.py
solution: self_play_data/solution.py
alternatives:
  - self_play_data/alternatives/loops.py
  - self_play_data/alternatives/deepcopy.py
hints:
  start:
    try: |
      history = [("s0", 1, "pi0"), ("s1", -1, "pi1")]
      state, player, pi = history[1]
      print(state, player, pi)
    see: |
      s1 -1 pi1
    next: "That's one history entry taken apart: the position, who was to move, and the policy. `to_examples` makes one `(state, pi, z)` from each entry. `z` is `result` seen from that entry's player: if X won (`result` is 1), what is `z` for this entry, where `player` is -1? Settle that, then write the loop."
  logic: |
    snapshot(board, player):
      build NEW rows, each cell turned to the mover's view
        (X keeps its signs, O's view flips them)
      → return the copy

    to_examples(history, result):
      for each (state, player, pi) in history:
        z = the game's result, seen from that player's side
          (+1 the mover won, -1 lost, 0 draw)
        → make (state, pi, z)
      → return the list, same order as history
  tools: |
    copy the board in the mover's view
      [f(x) for x in row]   builds a new list from row
      [... for row in rows] one new row per old row
      plain arithmetic: * multiplies, so -1 flips a sign
    one example per move
      for a, b, c in lst   unpacks three items per entry
      (a, b, c)            a tuple of three items
      plain arithmetic: *
  nudge: The net always sees the board from the side of the player to move. How do you turn the board into O's view? And when X wins, what did the game mean for the positions where O was to move?
  concept: "Multiplying every cell by the player gives the mover's view: for X (1) the board stays as it is, for O (−1) every piece flips sign. Building new rows with a comprehension makes it a real copy at the same time. The label uses the same trick: `result * player` is +1 where the mover went on to win, −1 where they lost, and 0 in a draw."
  example: |
    Trace `snapshot` on a row `[0, 1, -1]`:

    - `player = 1`: the row stays `[0, 1, -1]`.
    - `player = -1`: every cell flips sign, so the row becomes `[0, -1, 1]`. O's pieces now look like 1.
    - Change the live board afterwards: the snapshot must not change.

    Now `to_examples` on a 3-move history, `players = [1, -1, 1]`:

    - `result = 1` (X won): the labels are `z = [1, -1, 1]`.
    - `result = -1` (O won): `z = [-1, 1, -1]`.
    - `result = 0` (draw): `z = [0, 0, 0]`.

    Each tuple keeps its `state` and `pi` and drops the `player`. The order is `(state, pi, z)`.
  skeleton: |
    # snapshot
    return [[cell * ______ for cell in row] for row in ______]
    # to_examples
    return [(state, ___, result * ______) for state, player, pi in history]
  solution: |
    # snapshot
    return [[cell * player for cell in row] for row in board]
    # to_examples
    return [(state, pi, result * player) for state, player, pi in history]
on_solve: "Solved. One game at 30 simulations a move gives a dozen or so labelled positions (13 in the reference), alternating +1 and −1 back from the winner's last move. A real run plays thousands of these games, trains the net on them, and plays the next batch with the new net. Step 4 builds the opponent that measures how strong it got."
mistakes:
  - { file: self_play_data/mistakes/placeholders.py, fails: snapshot_is_copy, says: "`snapshot` still returns `...`" }
  - { file: self_play_data/mistakes/live_board.py, fails: snapshot_is_copy, says: "returns the live board itself" }
  - { file: self_play_data/mistakes/shared_rows.py, fails: snapshot_is_copy, says: "still the board's own rows" }
  - { file: self_play_data/mistakes/no_flip.py, fails: snapshot_view, says: "still shows O's pieces as −1" }
  - { file: self_play_data/mistakes/always_flip.py, fails: snapshot_view, says: "flips X's pieces" }
  - { file: self_play_data/mistakes/examples_placeholder.py, fails: example_shape, says: "`to_examples` still returns `...`" }
  - { file: self_play_data/mistakes/keeps_player.py, fails: example_shape, says: "The order is `(state, pi, z)`" }
  - { file: self_play_data/mistakes/z_is_result.py, fails: z_from_mover, says: "X's win is a loss" }
  - { file: self_play_data/mistakes/z_flipped.py, fails: z_from_mover, says: "Every z is flipped" }
  - { file: self_play_data/mistakes/draw_is_loss.py, fails: draw_is_zero, says: "a draw counted as a loss" }
  - { file: self_play_data/mistakes/no_game.py, fails: self_play_runs, says: "is missing" }
```

### The depth-4 minimax bot

```challenge
id: minimax
file: connect4.py
minutes: 40
start: { from: self_play_data }
setup:
  - say: To measure your agent you need a fixed opponent. This one scores a position by counting every line of four that only one player has pieces in.
    type_file: minimax/heuristic.py
    pause: 2.8
  - say: "negamax looks a few moves ahead. Each side picks the move that's best for itself, and one side's score is the other side's score, negated. Fill in how a move gets scored."
    type_file: minimax/stub.py
    pause: 1.2
tests: minimax/tests.py
solution: minimax/solution.py
alternatives:
  - minimax/alternatives/if_else.py
  - minimax/alternatives/try_finally.py
hints:
  start:
    try: |
      row = drop(puzzle, 5, 1)
      print(row, is_win(puzzle, row, 5))
      undo(puzzle, 5)
      print(puzzle[5])
    see: |
      5 True
      [0, -1, 1, 1, 1, 0, 0]
    next: "That's column 5 tried for X on the puzzle board: the piece landed in row 5 and made four, so that move scores `1000 + depth`. Then `undo` took it back and the row is as it was. For a column that doesn't win, the score comes from the opponent's best reply one level deeper, and that score is from the opponent's view."
  logic: |
    for each open column col:
      drop player's piece into col → keep the row
      did that piece complete four?
        yes → score = 1000 + depth   (sooner win, higher)
        no  → ask the opponent's best score, one level
              shallower (the opponent moves next)
              → flip its sign   (it's their view)
              → that is score
      take the piece back out of col
      (the stub keeps the best score and column)
  tools: |
    try the piece
      drop(board, c, p)    puts p's piece in c, returns its row
      is_win(board, r, c)  did the piece at r, c make four?
    score it
      a if cond else b     picks a or b
      negamax(board, d, p) (score, column) for p to move, d deep
      plain arithmetic: - flips a number's sign
    take it back
      undo(board, c)       takes the top piece out of c
  nudge: For each column, try it, score it, take it back. When the move doesn't win on the spot, its score comes from the opponent's best reply. Whose view is that score from?
  concept: "Drop the piece and keep its row. If is_win says it completed a four, the move scores 1000 + depth (more depth left means a sooner win). Otherwise call `negamax(board, depth - 1, -player)`, the opponent to move one level shallower. Its score is from the opponent's view, so negate it. Then undo the piece before the loop tries the next column."
  example: |
    Trace the starter position. X has the bottom row at columns 2, 3 and 4, and O has columns 1 and 3 (one piece each, O's second on top of column 3).

    - X to move, `depth = 1`: dropping in column 5 completes a four, so `score = 1000 + 1 = 1001`. `negamax` returns `(1001, 5)`.
    - O to move, `depth = 2`. Each column is dropped, scored, and undone:
      - Column 5 blocks X. It doesn't win, so X's best reply at `depth = 1` is scored by the heuristic as 5. Flipped: `-5`.
      - Every other column leaves column 5 open, so X's best reply is the win, 1001. Flipped: `-1001`.
    - The best of those is `-5`, so `negamax` returns `(-5, 5)`: O plays column 5.

    Check your own run: after `negamax` returns, the board should look exactly as before.
  skeleton: |
    row = drop(board, col, ______)
    score = 1000 + depth if is_win(board, row, col) else ___negamax(board, depth - ___, ______)[0]
    ______(board, col)
  solution: |
    row = drop(board, col, player)
    score = 1000 + depth if is_win(board, row, col) else -negamax(board, depth - 1, -player)[0]
    undo(board, col)
on_solve: "Solved. The bot takes a win that's one move away and blocks one, but it can't see past 4 moves, and every move it looks at up to 7⁴ = 2,401 positions. Next, watch self-play training learn to beat it."
mistakes:
  - { file: minimax/mistakes/placeholder.py, fails: board_restored, says: "`score` is still `...`" }
  - { file: minimax/mistakes/no_undo.py, fails: board_restored, says: "never takes it back out" }
  - { file: minimax/mistakes/no_win_check.py, fails: takes_the_win, says: "check whether that piece just won" }
  - { file: minimax/mistakes/flat_win.py, fails: takes_the_win, says: "should score 1000 + depth" }
  - { file: minimax/mistakes/no_negation.py, fails: blocks_the_threat, says: "negate it" }
  - { file: minimax/mistakes/same_player.py, fails: blocks_the_threat, says: "plays the same player again" }
```

### Train by self-play and play the bot

```replay
id: train_c4
intro: "The full AlphaZero loop on Connect 4: self-play with your search, a residual policy/value net in place of rollouts, train on the examples, repeat. It needs a GPU and most of a working day, so watch a run first."
log: replays/train-c4.jsonl
recorded:
  hardware: RTX 3090 + 8 CPU workers
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a log of your own run.
chart:
  title: Games lost to the depth-4 minimax bot, per 100
  flag_above: 40
  x_labels: [gen 0, gen 12]
  caption: "Generation 0 has a uniform policy and no idea of value, and loses 92 of 100. By generation 5 it wins more than it loses, and by 12 it wins 88. The bot can't see past 4 moves; the value head learns to."
summary:
  - { label: Win rate vs minimax-4, value: 6% → 88% }
  - { label: Self-play games, value: "6,000" }
  - { label: Training time, value: 7 h 50 min }
run_yourself: |
  # on your GPU machine, from the course repo
  python sidequests/train_c4.py --generations 12 --games-per-gen 500 \
    --sims 200 --workers 8
```

### Record your metrics

```metrics
id: record
intro: Your agent's win rate against the depth-4 bot, and your in-browser simulations per second. Both are compared with the reference run.
```

## Hints

- Mirror every training position left to right, and its π with it. Connect 4 is symmetric, so that doubles your data for free.
- Batch the net calls. Run many self-play games at once and evaluate their leaves together on the GPU; one leaf per call leaves the GPU idle.
- Desktop CPython runs this lesson's code about 2–3× faster than the browser's Pyodide. Two 64-bit integers as a bitboard, one per player, are faster still.
- Keep the minimax bot fixed. Its strength never changes, so its win rate is a fair ruler from one generation to the next.
- Self-play adds Dirichlet noise to the root's priors so the search keeps trying moves the net dislikes. Without it, early mistakes in the policy never get corrected.

## Done when

- [ ] is_win agrees with a full-board check over 200 random games {#win_detection auto=challenge:win_detection}
- [ ] Self-play games turn into (state, π, z) examples {#self_play auto=challenge:self_play_data}
- [ ] Your trained agent beats the depth-4 minimax bot in most games {#beats_minimax}
- [ ] Win rate and simulations per second recorded {#metrics_recorded auto=metrics}
