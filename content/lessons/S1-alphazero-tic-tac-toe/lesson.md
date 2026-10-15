---
id: "S1"
title: AlphaZero tic-tac-toe
summary: Build AlphaZero's search in miniature, with PUCT selection, a sign-flipping backup and visit-count policies, then play it against a perfect tic-tac-toe player.
phase: "S"
hours: 2
type: sidequest
status: ready
protected: false
cut: null
prerequisites:
  - "0.2"
papers:
  - title: Mastering Chess and Shogi by Self-Play (AlphaZero)
    authors: Silver et al.
    year: 2017
    url: https://arxiv.org/abs/1712.01815
quality_metric:
  id: draw_rate
  label: Draw rate vs perfect play
  unit: "%"
  note: A perfect player never loses, so a draw is the best result. Step 4's script with SIMS = 800 and GAMES = 20, or your self-play net's last eval.
  inputs:
    - { id: draw_rate, label: Draws vs perfect play (%), hint: "Step 4 with SIMS = 800, GAMES = 20" }
  compute: value
  reference: 100
  pass: { min: 90 }
systems_metric:
  id: self_play_games
  label: Self-play games to converge
  unit: count
  note: Self-play games until the net's first eval with no game lost to the perfect player. Fewer is better.
  inputs:
    - { id: self_play_games, label: Self-play games to converge, hint: From your self-play run's eval lines }
  compute: value
  reference: 1200
  pass: { max: 5000 }
quiz:
  - id: visits_not_q
    q: After 800 simulations from the root, move 4 has 610 visits and Q = 0.10, and move 0 has 12 visits and Q = 0.40. Which move does AlphaZero play?
    answer: 0
    options:
      - text: Move 4, the most visited
        why: PUCT keeps sending simulations to moves that keep looking good. 610 visits means the search came back to move 4 again and again after checking everything else. Move 0's Q rests on 12 noisy samples.
      - text: Move 0, because its Q is higher
        why: An average over 12 simulations is noisy. If move 0 were really better, PUCT would have moved visits to it, since its bonus stays large while its n is small.
      - text: Neither. Keep searching until Q and visits agree
        why: They rarely agree exactly, and the game needs a move. The visit count already folds in both how good a move looks and how sure the search is.
  - id: sign_flip
    q: A simulation reaches a position where X has just completed a line, so O is to move and has lost. backup gets value = −1. What gets added to w of the node for X's winning move?
    answer: 0
    options:
      - text: "+1"
        why: That node's w counts for the player who moved into it, X. The first flip turns O's −1 into X's +1 before it's added.
      - text: "−1"
        why: −1 is the result from O's view, the player to move at the leaf. The leaf's w counts for whoever moved into it, which is X.
      - text: "0, because terminal nodes don't keep values"
        why: They do. A terminal leaf is visited like any other, and its exact result is the most reliable value in the whole tree.
  - id: prior_role
    q: In this lesson every legal move gets the same prior P. What changes when P comes from a trained policy net instead?
    answer: 1
    options:
      - text: Nothing. P cancels out when every move uses it
        why: P scales each move's exploration bonus separately, so different priors change which unvisited move gets tried first.
      - text: The search tries the net's favourite moves first, so it needs fewer simulations
        why: The bonus c·P·√N / (1 + n) is largest for high-P moves, so early visits go where the net points. That's how S1's net matches 800 random-rollout simulations with 50.
      - text: P replaces Q in the score
        why: Q is still the average backed-up value. A net replaces the rollout that produces the value; the policy head only supplies P.
---

## Goal

Build the search at the heart of AlphaZero, small enough to check by hand. Write tic-tac-toe's rules, pick moves in the tree with **PUCT** (Q + c·P·√N / (1 + n)), **back up** each result with its sign flipping at every level, and turn the root's **visit counts** into a policy. Then play your search against a perfect player and count the draws. The tiny policy/value net that replaces random rollouts is the optional last step, run on your own CPU.

## Learn

```card
type: concept
id: four_steps
eyebrow: The search
title: One simulation, four steps.
visual:
  kind: bars
  max: 100
  bars:
    - { label: 25 sims, value: 65, text: "65%" }
    - { label: 100 sims, value: 75, text: "75%" }
    - { label: 400 sims, value: 90, text: "90%" }
    - { label: 800 sims, value: 100, text: "100%" }
body: |
  Each simulation **selects** down the tree by PUCT until it reaches a leaf, **expands** the leaf with one child per legal move, **evaluates** it (a random rollout here, a value net in AlphaZero) and **backs up** the result along the path it took. After many simulations, the most visited move at the root is the move to play. The bars show draws against a perfect player over 20 games with random rollouts: more simulations, fewer losses.
```

```card
type: fill_blank
id: puct_blanks
eyebrow: Fill the blanks
title: Complete the PUCT score.
code: |
  score = child.q() + c_puct * [[0]] * math.sqrt([[1]]) / (1 + [[2]])
tokens: [child.prior, node.prior, node.n, child.n, child.w]
answer: [child.prior, node.n, child.n]
tips:
  node.prior: That's the prior of the move into the *parent*. It's the same for every child, so it can't rank them.
  child.w: w is a running total that grows with every visit. The bonus should shrink as a move gets tried, which is what `child.n` does.
why_ok: "The bonus favours moves the prior likes (P), grows slowly as the parent is visited (√N), and fades as this child gets tried (1 + n)."
why_wrong: "P is the child's own prior, N is how often the parent was visited, and n is how often this child was."
hint: N is the parent's visit count; n is the child's.
```

```card
type: calc
id: puct_by_hand
eyebrow: Do the math
title: What's this child's PUCT score?
given:
  - { label: "Q (child's average value)", value: "0.30" }
  - { label: "P (child's prior)", value: "0.50" }
  - { label: "N (parent's visits)", value: "100" }
  - { label: "n (child's visits)", value: "24" }
  - { label: c_puct, value: "1.5" }
unit: ""
answer: { min: 0.59, max: 0.61 }
placeholder: "0.00"
why_ok: "0.30 + 1.5 × 0.5 × √100 / (1 + 24) = 0.30 + 0.30 = 0.60. Visit it 75 more times without Q changing and the parent's N grows to 175 too, so the bonus shrinks to 1.5 × 0.5 × √175 / 100 ≈ 0.099."
why_wrong: "The bonus is c · P · √N / (1 + n) = 1.5 × 0.5 × 10 / 25. Add Q to it."
hint: "√100 is 10, and 1 + 24 is 25."
```

```card
type: predict
id: backup_signs
eyebrow: Predict the output
title: The player to move at the leaf has won. What prints?
code: |
  path = [Node(1.0), Node(0.5), Node(0.5), Node(0.5)]   # root … leaf
  backup(path, 1.0)
  print([node.w for node in path])
options:
  - text: "[1.0, -1.0, 1.0, -1.0]"
    correct: true
    why: The leaf's w counts for whoever moved into it, the opponent of the winner, so it gets −1. Its parent's move was the winner's, +1, and so on up to the root.
  - text: "[-1.0, 1.0, -1.0, 1.0]"
    why: That adds before flipping, so the leaf gets +1. But value is for the player to move at the leaf, and the leaf's w is for the other player.
  - text: "[1.0, 1.0, 1.0, 1.0]"
    why: Without a flip, every move on the path looks good, for both players. A win for one side is a loss for the other.
  - text: "[0.0, 0.0, 0.0, -1.0]"
    why: That updates only the leaf. Every node on the path took part in this simulation, so all of them get the result.
hint: Start at the leaf. Its w is from the view of the player who moved into it, not the player to move there.
```

```card
type: reorder
id: one_simulation
eyebrow: Put it in order
title: Arrange one simulation.
lines:
  - "Walk down from the root, picking children by PUCT"
  - "Stop at a leaf: a position nobody has expanded yet"
  - "Add a child for every legal move, each with prior 1 / (number of moves)"
  - "Play a random rollout from the leaf to get a value"
  - "Back the value up the path, flipping its sign at every level"
shuffle: [3, 0, 4, 2, 1]
why_ok: "Select, expand, evaluate, back up. After all the simulations, play the root's most visited move."
why_wrong: "Each step needs the one before: you can't expand a leaf you haven't reached, or back up a value you haven't computed."
hint: The rollout needs a leaf to start from, and backup needs the rollout's value.
```

```card
type: quiz
ref: visits_not_q
```

```card
type: quiz
ref: sign_flip
```

```card
type: quiz
ref: prior_role
```

## Build steps

### Game rules: legal moves and the winner

```challenge
id: rules
file: alphazero_ttt.py
minutes: 20
start: { code: "import math\nimport random\n\n" }
setup:
  - say: A tic-tac-toe board is a list of nine cells. 1 is X, −1 is O, 0 is empty.
    type: "# A board is a list of 9 cells, left to right, top to bottom:\n#    0 | 1 | 2\n#    3 | 4 | 5\n#    6 | 7 | 8\n# 1 is X, -1 is O, 0 is empty. X moves first.\n\n"
    pause: 2.4
  - say: Three in a row wins, along any of these eight lines.
    type: "LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]\n\n\n"
    pause: 2.0
  - say: "Here's the stub. Your turn: write legal_moves and winner, then run the tests."
    type_file: rules/stub.py
    pause: 1.2
tests: rules/tests.py
solution: rules/solution.py
alternatives:
  - rules/alternatives/enumerate_all.py
  - rules/alternatives/loops.py
hints:
  nudge: Two questions about a board. Which cells are still empty? And has anyone filled a line? For winner, also think about what makes a game a draw rather than unfinished.
  concept: "A line wins when its three cells hold the same player *and* that player isn't 0, because three empty cells are equal too. Return that cell's value, so X's line gives 1 and O's −1. Only when no line wins: a board with no 0 left is a draw, 0, and anything else is still going, None."
  skeleton: |
    # legal_moves
    return [i for i in range(9) if ________]
    # winner
    for a, b, c in LINES:
        if board[a] != ___ and board[a] == board[b] == ________:
            return ________
    return 0 if ________ else None
  solution: |
    # legal_moves
    return [i for i in range(9) if board[i] == 0]
    # winner
    for a, b, c in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0 if 0 not in board else None
on_solve: Solved. Your rules agree with every one of the 5,478 positions a real game can reach. Step 2 is unlocked. Build the tree's nodes and the rule that walks down them.
mistakes:
  - { file: rules/mistakes/placeholders.py, fails: legal_moves, says: "`legal_moves` still returns `...`" }
  - { file: rules/mistakes/occupied_cells.py, fails: legal_moves, says: "those cells are already taken" }
  - { file: rules/mistakes/cell_values.py, fails: legal_moves, says: "the *contents* of the empty cells" }
  - { file: rules/mistakes/range8.py, fails: legal_moves, says: "`range(8)`" }
  - { file: rules/mistakes/winner_placeholder.py, fails: ongoing_is_none, says: "`winner` still returns `...`" }
  - { file: rules/mistakes/empty_line_wins.py, fails: ongoing_is_none, says: "all of them empty" }
  - { file: rules/mistakes/always_draw.py, fails: ongoing_is_none, says: "A draw needs a full board" }
  - { file: rules/mistakes/draw_inverted.py, fails: ongoing_is_none, says: "upside down" }
  - { file: rules/mistakes/returns_false.py, fails: ongoing_is_none, says: "returns False" }
  - { file: rules/mistakes/full_line_wins.py, fails: ongoing_is_none, says: "every cell taken, but not by one player" }
  - { file: rules/mistakes/returns_true.py, fails: wins_for_both, says: "*who* won" }
  - { file: rules/mistakes/always_x.py, fails: wins_for_both, says: "Which cell's value tells you who owns the line?" }
  - { file: rules/mistakes/only_x.py, fails: wins_for_both, says: "does your check work for −1 too?" }
  - { file: rules/mistakes/no_draw.py, fails: full_board_draw, says: "would never end" }
  - { file: rules/mistakes/draw_first.py, fails: full_board_draw, says: "check the lines first" }
```

### Pick a child with PUCT

```challenge
id: select_child
file: alphazero_ttt.py
minutes: 25
start: { from: rules }
setup:
  - say: "Each node in the search tree is one position, reached by one move. It keeps that move's prior P, its visit count n and its total value w."
    type_file: select_child/node.py
    pause: 3.0
  - say: "Selection walks down the tree. At each node it takes the child with the highest PUCT score: Q, how well the move has done so far, plus a bonus that's big for moves the prior likes and shrinks as a move gets tried. Fill in the score."
    type_file: select_child/stub.py
    pause: 1.2
tests: select_child/tests.py
solution: select_child/solution.py
alternatives:
  - select_child/alternatives/split_terms.py
  - select_child/alternatives/inline_q.py
  - select_child/alternatives/max_key.py
hints:
  nudge: The score has two parts. One is what the move has earned so far, the other a bonus for moves that haven't been tried much. Which fields of the node and the child feed each part?
  concept: "Q is `child.q()`, the average value per visit (0 for an unvisited child). The bonus is c_puct × P × √N / (1 + n): P is `child.prior`, N is the parent's `node.n` and n is `child.n`. It grows slowly as the parent gets visited, fades as this child does, and the `1 +` keeps an unvisited child from dividing by zero."
  skeleton: |
    score = child.____() + c_puct * child.______ * math.sqrt(____.n) / (1 + _____.n)
  solution: |
    score = child.q() + c_puct * child.prior * math.sqrt(node.n) / (1 + child.n)
on_solve: "Solved. With the parent at 10 visits, move 4's bonus (1.5 × 0.5 × √10 / 1 ≈ 2.4) beats move 0's Q of 0.1 plus its bonus of 0.24, so the search tries move 4 next. Step 3 wires your PUCT into a whole search."
mistakes:
  - { file: select_child/mistakes/placeholder.py, fails: explores_unvisited, says: "`score` is still `...`" }
  - { file: select_child/mistakes/sqrt_child_n.py, fails: explores_unvisited, says: "the parent's visit count" }
  - { file: select_child/mistakes/divide_by_n.py, fails: explores_unvisited, says: "division by zero" }
  - { file: select_child/mistakes/q_by_hand.py, fails: explores_unvisited, says: "division by zero" }
  - { file: select_child/mistakes/no_prior.py, fails: prior_orders_unvisited, says: "Where does P go in the score?" }
  - { file: select_child/mistakes/no_q.py, fails: exploits_good_moves, says: "knows how well a move has gone" }
  - { file: select_child/mistakes/negated_q.py, fails: exploits_good_moves, says: "the one with the *lowest* Q" }
  - { file: select_child/mistakes/total_w.py, fails: average_not_total, says: "per visit" }
  - { file: select_child/mistakes/fixed_c.py, fails: uses_c_puct, says: "using the `c_puct` it's given" }
  - { file: select_child/mistakes/sqrt_whole.py, fails: matches_puct, says: "PUCT picks move" }
```

### Back up the result

```challenge
id: backup
file: alphazero_ttt.py
minutes: 25
start: { from: select_child }
setup:
  - say: "Here's the whole search. Each simulation walks down with your select_child, gives a new leaf one child per legal move with a uniform prior, and plays a random game from there to get a value. AlphaZero asks a value net instead."
    type_file: backup/search.py
    pause: 3.4
  - say: "The last piece sends that value back up the path. Careful whose view it's from: the levels alternate between X's moves and O's."
    type_file: backup/stub.py
    pause: 1.2
tests: backup/tests.py
solution: backup/solution.py
alternatives:
  - backup/alternatives/index_loop.py
  - backup/alternatives/sign_from_depth.py
hints:
  nudge: Every node on the path took part in this simulation. What does each one gain? And is the result good or bad for the player who made the move into it?
  concept: "Walk the path from the leaf up with `reversed(path)`. value starts out as the result for the player *to move* at the leaf, while the leaf's w counts for the player who moved *into* it, their opponent. So flip first, then add: `value = -value`, one more visit, `w` plus value. The next node up was the other player's move, so one flip per step keeps every node in its own player's view."
  skeleton: |
    for node in ________(path):
        value = ______
        node.n += _
        node.w += _____
  solution: |
    for node in reversed(path):
        value = -value
        node.n += 1
        node.w += value
on_solve: "Solved. Almost every simulation went to the winning move, 2 (186 of 200 in the reference). Once select_child sees its Q of +1, the bonus of the other moves never catches up. Step 4 turns visit counts into the move to play and into a training target."
mistakes:
  - { file: backup/mistakes/placeholder.py, fails: counts_visits, says: "still the `...` placeholder" }
  - { file: backup/mistakes/leaf_only.py, fails: counts_visits, says: "Only the leaf got a visit" }
  - { file: backup/mistakes/no_visits.py, fails: counts_visits, says: "n is still 0 everywhere" }
  - { file: backup/mistakes/no_flip.py, fails: signs_alternate, says: "Flip the sign at each step up" }
  - { file: backup/mistakes/flip_after.py, fails: signs_alternate, says: "Flip before you add" }
  - { file: backup/mistakes/root_first.py, fails: signs_alternate, says: "The sign you know is the leaf's" }
  - { file: backup/mistakes/assign_w.py, fails: adds_up, says: "w is a running total" }
```

### Visit counts into a policy, then play a perfect player

```challenge
id: visit_policy
file: alphazero_ttt.py
minutes: 30
start: { from: backup }
setup:
  - say: "To check your search, play it against a perfect player. This one solves tic-tac-toe outright by searching every game to the end, which is cheap on a 9-cell board."
    type_file: visit_policy/perfect.py
    pause: 2.8
  - say: "AlphaZero plays the most visited move, and trains its policy head on the visit distribution. Write visit_policy, then watch your search play."
    type_file: visit_policy/stub.py
    pause: 1.2
tests: visit_policy/tests.py
solution: visit_policy/solution.py
alternatives:
  - visit_policy/alternatives/loop_and_normalise.py
  - visit_policy/alternatives/child_visits.py
hints:
  nudge: Start from each cell's visit count, with 0 for cells that have no child. How do you turn nine counts into nine probabilities? And what should temperature 0 mean?
  concept: "Collect n per cell (0 where there's no child), raise each to `1 / temperature`, and divide by the total so the nine add up to 1. Temperature 1 leaves the counts as they are; 0.5 squares them, which sharpens the policy. At temperature 0 that power would be `1 / 0`, so handle it first: 1.0 on the most visited cell, 0 everywhere else."
  skeleton: |
    visits = [root.children[m].n if m in root.children else ___ for m in range(___)]
    if temperature == 0:
        best = visits.index(________)
        return [1.0 if m == best else 0.0 for m in range(9)]
    powered = [n ** (________) for n in visits]
    total = ________
    return [x / total for x in powered]
  solution: |
    visits = [root.children[m].n if m in root.children else 0 for m in range(9)]
    if temperature == 0:
        best = visits.index(max(visits))
        return [1.0 if m == best else 0.0 for m in range(9)]
    powered = [n ** (1 / temperature) for n in visits]
    total = sum(powered)
    return [x / total for x in powered]
on_solve: "Solved. The reference search draws all 10 games. The perfect player can't lose, so a draw is the best anyone can do. Now find where the search breaks: set SIMS = 25 and GAMES = 20 and run again (the reference loses 7 of 20), then SIMS = 100 (5 lost), then 800 (none). With a uniform prior and random rollouts it takes hundreds of simulations a move. The next step shows what a trained net does to that number."
mistakes:
  - { file: visit_policy/mistakes/placeholder.py, fails: nine_cells, says: "still the `...` placeholder" }
  - { file: visit_policy/mistakes/one_per_child.py, fails: nine_cells, says: "one per child" }
  - { file: visit_policy/mistakes/divide_by_root_n.py, fails: follows_visits, says: "You divided by `root.n`" }
  - { file: visit_policy/mistakes/from_q.py, fails: follows_visits, says: "*visit* distribution" }
  - { file: visit_policy/mistakes/power_temperature.py, fails: temperature, says: "*flatter* than at 1" }
  - { file: visit_policy/mistakes/times_temperature.py, fails: temperature, says: "temperature has no effect" }
  - { file: visit_policy/mistakes/zero_crash.py, fails: temperature, says: "division by zero" }
  - { file: visit_policy/mistakes/returns_move.py, fails: temperature, says: "a move" }
  - { file: visit_policy/mistakes/no_games.py, fails: script_runs, says: "are missing" }
```

### Self-play: a tiny net replaces the rollouts

```replay
id: selfplay_ttt
intro: "The optional half of AlphaZero: a small policy/value net that learns from its own games. It runs on a laptop CPU in a few minutes, outside the browser. Watch a run first so you know what convergence looks like."
log: replays/selfplay-ttt.jsonl
recorded:
  hardware: 1 laptop CPU core
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a log of your own run.
chart:
  title: Games lost to perfect play, per 100
  flag_above: 10
  x_labels: [0 games, 2000 games]
  caption: With a trained net and only 50 simulations a move, the search stops losing after about 1,200 self-play games. Random rollouts needed around 800 simulations a move for the same result.
summary:
  - { label: Simulations per move, value: 50 with the net · ~800 with rollouts }
  - { label: Self-play games to converge, value: "1,200" }
  - { label: Draws vs perfect play, value: 100 / 100 }
run_yourself: |
  # on your machine, from the course repo (Python + NumPy, no GPU)
  python sidequests/selfplay_ttt.py --games 2000 --sims 50 --eval-every 200
```

### Record your metrics

```metrics
id: record
intro: Your draw rate against perfect play, and how many self-play games your net needed. Both are compared with the reference run.
```

## Hints

- If your search loses, raise SIMS before you suspect the code. Random rollouts play badly on purpose, so a few hundred of them still miss some forced threats.
- c_puct trades exploring for exploiting. Try 0.5 and 4 with SIMS = 200 and watch the draw count move.
- The perfect player is negamax over every position, cached with `lru_cache`: about 5,500 positions, solved once, then looked up.
- In real AlphaZero one net call per simulation gives both P (policy head) and the leaf value (value head). There are no rollouts at all.
- During self-play AlphaZero also mixes Dirichlet noise into the root's priors, so even moves the net dislikes get tried now and then. S1's uniform prior doesn't need it.

## Done when

- [ ] Your MCTS takes a winning move and blocks a losing one {#search_works auto=challenge:backup}
- [ ] At 800 simulations a move, your search draws all 20 games against perfect play {#draws_perfect}
- [ ] Draw rate and self-play games recorded {#metrics_recorded auto=metrics}
