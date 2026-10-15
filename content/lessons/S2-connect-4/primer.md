---
gist: Connect 4 is S1's search on a game too big to brute-force, so speed, good training data and a fixed opponent to measure against start to matter. This lesson builds all three.
minutes: 16
---

## What this lesson is for {#why}
> You'll take S1's search to Connect 4: a fast win check, rollouts that don't copy the board, self-play training data, and a minimax bot to measure progress.

In S1 the whole game fit in memory and a perfect player could check your work. Connect 4 is different. It has about 4.5 trillion positions, so nobody solves it on the fly, and every simulation costs more because a random game can run for dozens of moves.

That changes what's worth your time. You'll write four pieces:

1. **A win check** that looks only around the piece just dropped.
2. **Fast rollouts** that play on one board and take their moves back, instead of copying.
3. **Self-play data**: turning a finished game into training examples for a network.
4. **A minimax bot** that looks 4 moves ahead, as a fixed opponent.

Then you watch a recorded training run learn to beat that bot. This primer assumes you've done S1; it reuses PUCT, backup and visit policies from there without explaining them again.

## The board in code {#board}
> The board is 6 rows of 7 cells. Row 0 is the top, so a dropped piece lands in the highest-numbered empty row of its column.

The board is a list of 6 lists, one per row, each with 7 cells. As in S1, `1` is X, `-1` is O and `0` is empty. `show` prints X and O:

```
row 0  . . . . . . .
row 1  . . . . . . .
row 2  . . . . . . .
row 3  . . . . . . .
row 4  O O . O . . .
row 5  X X X X . . .
       0 1 2 3 4 5 6
```

`board[5][2]` is row 5, column 2: the bottom row, third cell. A move is a column number. `drop(board, col, player)` scans the column from the bottom up, puts the piece in the first empty cell and returns that row. A column is a legal move while its top cell, `board[0][col]`, is empty.

## Checking for a win around the last piece {#win}
> A new four must pass through the piece just dropped, so count matching pieces outwards from it in four directions.

The board was checked after every earlier move, and only one cell has changed since. So any new four includes the new piece. There are four directions a line can run:

| (dr, dc) | direction |
|---|---|
| (0, 1) | across |
| (1, 0) | down |
| (1, 1) | diagonal, falling to the right |
| (1, −1) | diagonal, rising to the right |

`run_length` walks from the new piece one step of (dr, dc) at a time and counts how many cells in a row hold the same player. It stops at the board's edge or at the first cell that isn't theirs. It doesn't count the starting piece.

`is_win` then adds the run in each direction and the run in the opposite direction, plus 1 for the piece itself. For the board above, X just dropped into column 2:

```
right of (5, 2): X at col 3 → 1
left of (5, 2):  X, X → 2
total: 1 + 1 + 2 = 4  → a win
```

Counting only one way would miss this four, because the new piece landed in the middle of it. Fives count too: filling the gap in `X X _ X X` makes five in a row, and five contains a four, so the test is "4 or more".

## The edge trap: negative indexes {#edges}
> In Python `row[-1]` is the last item, not an error, so a walk that steps off the left or top edge silently wraps round to the other side.

Java throws `ArrayIndexOutOfBoundsException` for index −1. Python doesn't: a negative index counts from the end. Walk left from column 0 on this bottom row:

```
row = X . . . . X X
c = -1 → row[-1] is X (col 6)
c = -2 → row[-2] is X (col 5)
c = -3 → row[-3] is .  stop
count = 2, but it should be 0
```

Your run has wrapped round the edge and found pieces that aren't neighbours. So a walk has to check both bounds, `0 <= r` and `r < ROWS`, and the same for columns. Going off the right or bottom edge does raise an `IndexError`, which at least tells you something is wrong. The left and top edges fail silently, which is worse.

## Why speed matters now {#speed}
> MCTS gets stronger with more simulations, and each Connect 4 simulation plays a whole random game, so simulations per second becomes the number to watch.

A tic-tac-toe rollout is at most 9 moves. A Connect 4 rollout is often 20 or more, and each move needs a win check. The search also needs more simulations to say anything useful, because the game is bigger and the random results are noisier.

Step 2 gives you a slow rollout to race against yours. It copies the whole board first (`copy.deepcopy`), then after every move checks all 42 cells for a four. Measured in the browser, over 300 simulations from the empty board:

| rollout | simulations per second |
|---|---|
| copy, then check every cell | about 1,400 |
| play on the board, check the last piece, undo | about 8,200 |

Your numbers will differ with your machine, but the gap is about 5×. That gap is the lesson's systems metric: the rollout line from step 2's speed test, with 3,000 a second or more to pass.

## Play, then undo {#undo}
> Instead of copying the board, play each move on it, remember the column, and take the moves back out in reverse order when you're done.

Think of the moves as a stack, like `Deque<Integer>` used as a stack in Java. Every move pushes its column. When the rollout ends, pop them all and undo each one.

**Undo** takes the *top* piece out of a column, the one dropped last. `drop` fills a column from the bottom up, so the top piece is the first non-empty cell when you scan from row 0 downwards:

```
column 2, top to bottom:
row 3: X   ← undo removes this
row 4: O
row 5: X
```

Common slips the tests catch: taking out the bottom piece (the oldest one), clearing the whole column, or forgetting to undo at all. That last one leaves pieces behind, and since the search reuses the same board for every simulation, they pile up until a column is full and `drop` fails.

**The result** is for the player the rollout *started* with, not "who won". If O starts a rollout and wins, the rollout returns +1, because MCTS needs the value from the view of the player to move at the leaf. And if the board fills with no four, the rollout has to stop and return 0. Asking for a random choice from an empty list of moves crashes.

## Self-play data: state, π and z {#data}
> Each move of a self-play game becomes one training example: the position as the mover sees it, the search's visit policy π, and the game's final result z from the mover's view.

In self-play the search plays both sides. At every move the code stores three things, and when the game ends it labels each of them:

| part | what it is |
|---|---|
| state | the board, as the player to move sees it |
| π | the visit policy over the 7 columns |
| z | the final result for that player: +1, −1 or 0 |

**The mover's view.** Multiply every cell by the player to move. For X (1) the board is unchanged. For O (−1) every piece flips sign, so O's own pieces show as 1. The network then only ever sees "my pieces" and "their pieces", and one set of weights plays both colours.

**A real copy.** The game keeps playing on the same board, so a stored state that shares its rows with the live board changes as the game goes on. By the end, every stored state would show the final position. Each state needs its own new rows.

**The label.** Say X wins a 5-move game. The movers were X, O, X, O, X:

| move | mover | z |
|---|---|---|
| 1 | X | +1 |
| 2 | O | −1 |
| 3 | X | +1 |
| 4 | O | −1 |
| 5 | X | +1 |

In a draw every z is 0, not −1 for both sides. Getting z wrong teaches the value head the opposite of the truth, and nothing crashes, so the tests check it carefully.

## Variety in self-play {#explore}
> The first 8 moves of each self-play game are sampled from the visit counts, so games open differently and the data covers more positions.

If self-play always played the most visited move, many games would start the same way and the network would see the same few openings again and again. So the first 8 moves are drawn at temperature 1, in proportion to the visits, and after that the game plays the search's best move (temperature 0). The training run in the replay also adds random noise to the root's priors, which AlphaZero does too, so the search keeps trying moves the network dislikes.

## A fixed opponent: the depth-4 minimax bot {#minimax}
> To see whether training is working you need an opponent whose strength never changes. A minimax search that looks 4 moves ahead is cheap and steady.

This is negamax from S1, but cut off after 4 moves, because the whole tree is far too big. At the cut-off the bot can't know the result, so it estimates it with a **heuristic**: a hand-written score for a position.

The heuristic looks at every **window**, a line of four cells. There are 69 of them on a 6×7 board: 24 across, 21 down and 12 on each diagonal. A window that only one player has pieces in is a possible four for them. More pieces in it is worth more:

| pieces in a window only you're in | points |
|---|---|
| 1 | 1 |
| 2 | 4 |
| 3 | 16 |

The position's score is your points minus the opponent's.

For each move the bot drops the piece, scores it, and takes it back out. A move that completes a four scores 1000 plus the depth left, so a sooner win scores higher. Otherwise the move is worth minus the opponent's best score one level deeper, which is the same sign flip as S1's backup. With 7 moves a turn, 4 moves ahead is up to 7 × 7 × 7 × 7 = 2,401 positions per move.

The tests check that the bot takes a win that's one move away and blocks one, that it leaves the board as it found it, and that the recursive call is the *other* player's turn and gets negated.

## The training run {#replay}
> The full AlphaZero loop runs on a GPU for most of a working day, so you watch an illustrative recording of it and read off what healthy training looks like.

The replay step shows the whole loop: self-play with the search, a small residual network in place of rollouts, train on the examples, repeat for 12 generations of 500 games each. The recording is **illustrative**, sample data written for the course and labelled that way, not a real run.

What healthy looks like:

| generation | games lost to the bot, of 100 |
|---|---|
| 0 (untrained) | 92 |
| 5 | 38 |
| 12 | 10 |

The untrained network has no idea which moves are good, so it loses almost every game. Policy and value loss fall each generation, and by the end the agent wins 88 games of 100. The bot can't see past 4 moves; the value head learns to judge positions beyond that. Your quality metric is your own agent's win rate against the bot, with 60% or more to pass.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Board**: 6 rows of 7 cells, row 0 at the top. `1` is X, `-1` is O, `0` is empty.
- **drop / undo**: put a piece in the lowest empty cell of a column / take the top piece back out.
- **Legal move**: a column whose top cell is empty.
- **run_length**: how many cells in a row, walking one way from a piece, hold the same player.
- **is_win**: whether the piece just dropped is part of four or more in a row.
- **Direction (dr, dc)**: the step for one line direction, like (0, 1) for across.
- **Negative index**: in Python, `row[-1]` is the last item. A walk off the left or top edge wraps round silently.
- **Rollout**: random play to the end of the game. Here it plays on the board itself and undoes its moves.
- **Simulations per second (sims/s)**: how fast the search runs. The lesson's systems metric.
- **Self-play**: the search playing both sides to make training data.
- **State / snapshot**: a copy of the board from the mover's side, own pieces 1.
- **π (pi)**: the visit policy over the 7 columns.
- **z**: the final result from the view of the player to move in that state.
- **Example**: one (state, π, z) triple.
- **Temperature**: how sharp the policy is when choosing a move. 1 samples by visits, 0 plays the most visited move.
- **Dirichlet noise**: randomness mixed into the root's priors during self-play.
- **Minimax / negamax**: looking ahead with each side picking its best move. Negamax flips the sign at each level.
- **Depth**: how many moves ahead the bot looks. The bot uses 4.
- **Heuristic**: a hand-written estimate of how good a position is.
- **Window**: a line of four cells. A 6×7 board has 69.
- **Policy head / value head**: the network's two outputs, trained toward π and z.
- **Generation**: one round of self-play games followed by training.
- **Win rate vs depth-4 minimax**: wins out of 100 games against the bot. The lesson's quality metric.
- **Replay (illustrative)**: a recorded run you watch in the lesson. Illustrative means sample data, not a real recording.
