---
gist: A game is a tree of positions too big to search completely, so AlphaZero samples it with Monte Carlo tree search, guided by a rule called PUCT. This lesson builds that search for tic-tac-toe.
minutes: 18
---

## What this lesson is for {#why}
> You'll build the search at the heart of AlphaZero, small enough to check by hand, and play it against a player that never loses.

AlphaZero is the program that taught itself chess, shogi and Go from nothing but the rules, and then beat the best existing programs. It has two parts: a **search** that looks ahead through possible moves, and a **neural network** that tells the search which moves look promising and who is winning. This sidequest is about the search. The network shows up at the end as an optional step.

Tic-tac-toe is the right size to learn on. It's small enough that a computer can play it perfectly, so you can check your search against the truth. It's also big enough that the search ideas matter.

You'll write four pieces, one per challenge:

1. **The rules**: which moves are legal, and who has won.
2. **Selection**: how the search decides which move to look at next (PUCT).
3. **Backup**: how a result travels back up the tree.
4. **Visit policy**: how the search turns its work into a move.

The only earlier lesson you need is 0.2 (micrograd), and only for Python. Everything else is explained below. S2 then takes the same search to Connect 4.

## Games as trees {#trees}
> Every position is a node, every legal move is an edge to a new position, and a finished game is a leaf.

The board in this lesson is a list of 9 cells, numbered left to right, top to bottom. `1` is X, `-1` is O, `0` is empty:

```
 0 | 1 | 2
 3 | 4 | 5
 6 | 7 | 8
```

Here's a position with X to move:

```
 X | X | .
 O | O | .
 . | . | .
```

As a list that's `[1, 1, 0, -1, -1, 0, 0, 0, 0]`. The legal moves are the empty cells: 2, 5, 6, 7 and 8. Each one leads to a new position, and each of those has its own moves. Drawn out, that's a **tree**, the same structure as a Java class with a `List<Node> children` field:

```
        [this position]
     /   |    |    |   \
   2     5    6    7    8
   |    /|\  ...
 X wins  ...
```

Playing 2 ends the game: X has the top row. A game ends when someone owns a whole row, column or diagonal (there are 8 such **lines**), or when the board is full with no line, which is a **draw**. Your first challenge is exactly this: `legal_moves` lists the empty cells and `winner` returns `1` or `-1` for a winner, `0` for a draw, and `None` while the game is still going.

Two mistakes the tests look for. Three *empty* cells are equal to each other, so "all three cells match" isn't enough on its own; the cells also have to be non-empty. And a board with empty cells left isn't a draw yet, so `0` and `None` have different meanings.

## Why you can't just search everything {#branching}
> The number of positions grows by a factor of the branching number at every move, which is fine for tic-tac-toe and hopeless for chess.

The **branching factor** is roughly how many legal moves a position has. Look `d` moves ahead with branching factor `b` and you have about `b` × `b` × … (`d` times) = `b^d` positions to look at.

| Game | Moves per turn | Typical length |
|---|---|---|
| Tic-tac-toe | 9 at most | 9 moves |
| Connect 4 | 7 at most | up to 42 |
| Chess | about 35 | about 80 |
| Go | about 250 | about 150 |

Tic-tac-toe has only 5,478 different positions a real game can reach, so you can search all of them. That's how the perfect player in step 4 works. Chess at 35 moves a turn gives 35 × 35 = 1,225 positions two moves ahead and about 1.8 billion six moves ahead. No computer searches a chess game to the end.

So a strong program has to choose which branches are worth looking at. That's the problem Monte Carlo tree search solves.

## Perfect play with minimax {#minimax}
> When the whole tree fits, score every finished game and let each player pick the best move for themselves, from the leaves upwards.

At a finished game the score is known: +1 if the player to move has won, −1 if they've lost, 0 for a draw. One level up, the player to move picks the child that's best for *them*. A child that's good for the opponent is bad for you, so you flip its sign:

```
my score = max over my moves
           of (−opponent's score)
```

This flip trick is called **negamax**. It's recursion, like walking a file tree in Java: score the children first, then combine. Step 4 gives you a perfect player built this way, with `lru_cache` so each of the 5,478 positions is solved only once.

A perfect tic-tac-toe player never loses. Two perfect players always draw. So against it, a **draw is the best result possible**, and your search's draw rate is its score.

## Monte Carlo: guessing by playing randomly {#rollouts}
> When you can't search to the end, play random moves until the game ends, and treat the result as a rough guess of how good the position was.

A **rollout** plays random legal moves for both sides until someone wins or the board fills. One rollout tells you little. A position where you'd win 70% of random games still loses some of them. But the *average* over many rollouts is a usable estimate of how good the position is. "Monte Carlo" is the general name for estimating something by averaging random samples.

In S1 a rollout is the search's only judge of a position. AlphaZero replaces it with the network's **value head**, which predicts the result directly in one call instead of playing a game out. That swap comes back at the end of this primer.

Random rollouts have a weakness: random players miss threats. If X threatens to win next move, a random O blocks it only some of the time, so the rollouts make the position look less dangerous for O than it is. The search fixes that by looking deeper into the positions that matter, which is the next idea.

## The search tree: four steps per simulation {#mcts}
> Each simulation walks down the tree, adds one new position, scores it with a rollout and sends the score back up. Many simulations later, the most visited move is the move to play.

MCTS keeps a tree of positions it has looked at. Each node stores:

| Field | Meaning |
|---|---|
| `prior` (P) | how much we liked the move into this node before searching |
| `n` | how many simulations passed through it |
| `w` | the total of their results |
| `q()` (Q) | the average result, `w / n` (0 when `n` is 0) |
| `children` | move → child node |

One **simulation** has four steps:

1. **Select.** Start at the root (the current position). While the node has children, step to the child with the best score (next section).
2. **Expand.** You've reached a **leaf**, a node with no children yet. Give it one child per legal move. In S1 every child gets the same prior, 1 divided by the number of moves.
3. **Evaluate.** Play a rollout from the leaf to get a value. If the leaf is already a finished game, use its real result.
4. **Back up.** Walk the path back to the root, adding one visit and the value to every node on it.

After, say, 200 simulations, the root's children have been visited different numbers of times. The search spent its visits on the moves that kept looking good, so the most visited child is the move to play.

In the position from earlier (X to move, X X . / O O .), the reference search sends 186 of 200 simulations to move 2, the winning move. Once it finds that move 2 wins every time, its Q is +1 and nothing else can compete.

## Choosing a child: UCB and PUCT {#puct}
> Each child's score is its average result Q plus an exploration bonus that's large for moves the search hasn't tried much and shrinks as they get tried.

If selection only picked the best Q, the search would lock onto whichever move got lucky in its first rollout and never look at the others. If it picked at random, it would waste time on bad moves. You want both: mostly **exploit** the moves that look good, but keep **exploring** the ones you know little about.

The classic answer is **UCB** (upper confidence bound): add a bonus to Q that grows with how often the parent was visited and shrinks with how often this child was. AlphaZero uses a variant called **PUCT**, which also weights the bonus by the prior P:

```
score = Q + c · P · √N / (1 + n)
```

- **Q**: the child's average result so far.
- **P**: the child's prior.
- **N**: the *parent's* visit count.
- **n**: the *child's* visit count.
- **c** (`c_puct`): how much to explore. S1 uses 1.5.

A worked example. The parent has been visited N = 10 times. Two children, both with prior P = 0.5:

| | n | w | Q | bonus | score |
|---|---|---|---|---|---|
| move 0 | 9 | 0.9 | 0.10 | 1.5 × 0.5 × 3.16 / 10 = 0.24 | 0.34 |
| move 4 | 0 | 0 | 0.00 | 1.5 × 0.5 × 3.16 / 1 = 2.37 | 2.37 |

(√10 ≈ 3.16.) Move 4 has never been tried, so its bonus is huge and the search tries it next. Each visit to move 4 makes its `1 + n` bigger and its bonus smaller, until its Q has to earn its place.

Things to get right, all of which the tests check:

- **N is the parent's count, n is the child's.** With the child's own n under the square root, an unvisited child gets a bonus of 0 and is never tried.
- **The `1 +` matters.** Dividing by n alone divides by zero for every unvisited child.
- **Q is an average, not a total.** A move visited 80 times has a big w just because it was visited a lot.
- **Use the `c_puct` you're given.** With c = 0 the bonus disappears and selection is pure Q.

## Whose point of view? Backing up with alternating signs {#backup}
> A result that's good for one player is bad for the other, so the value flips sign at every level on the way back up.

Tic-tac-toe has two players taking turns, so the levels of the tree alternate: X's moves, then O's, then X's. Each node's `w` is kept from the view of the **player who made the move into that node**. That way, when a player is choosing among children, a higher Q always means "better for me".

The rollout's value is from the view of the **player to move at the leaf**. The player who moved *into* the leaf is their opponent, so the value has to flip before it's added to the leaf. Then it flips again for the node above, and so on to the root.

Here's a path of four nodes after a rollout where the player to move at the leaf **won** (value +1):

| Node | Whose move led here | Added to w |
|---|---|---|
| leaf | the loser's | −1 |
| parent | the winner's | +1 |
| grandparent | the loser's | −1 |
| root | the winner's | +1 |

Every node also gets `n += 1`, because this simulation passed through all of them. Three classic slips, each of which the tests can see:

- **No flip.** Every node gets +1, so every move looks good for both players.
- **Adding before flipping.** The leaf gets +1 when it should get −1, and every sign is reversed.
- **Walking from the root down.** The first flip lands on the root, so whether the leaf ends up right depends on how long the path is. Start from the leaf, where you know the sign.

## Visit counts become the move and the policy {#policy}
> Normalise the root's visit counts into probabilities: that's both the move to play and the training target for a policy network.

After the simulations, turn the visits into a **policy**: one probability per cell, 9 in all. Cells with no child (taken cells) get 0. Say the children of cells 0, 2 and 4 have 10, 30 and 60 visits:

| cell | 0 | 2 | 4 | others |
|---|---|---|---|---|
| visits | 10 | 30 | 60 | 0 |
| policy | 0.10 | 0.30 | 0.60 | 0 |

Divide by the children's total, 100, not by the root's `n`. The root's first simulation expanded it without visiting any child, so the root's `n` is one more than its children's visits added up. Dividing by it gives a policy that sums to 0.99.

**Temperature** controls how sharp the policy is. Raise every count to the power `1 / temperature` before normalising:

| temperature | power | policy for 10, 30, 60 |
|---|---|---|
| 1 | n¹ | 0.10, 0.30, 0.60 |
| 0.5 | n² | 0.02, 0.20, 0.78 |
| 0 | (the limit) | 0, 0, 1 |

Temperature 0 means "all the probability on the most visited move". You can't compute `1 / 0`, so it's a special case. Temperature 1 is used early in self-play games for variety; temperature 0 is used to play your best.

Why visits and not Q? Q on a rarely visited move is an average of a few noisy rollouts. Visits already combine "looks good" with "the search is confident", because PUCT keeps returning to moves that hold up.

## How good is the search? {#draw-rate}
> More simulations per move means fewer losses against perfect play. With random rollouts it takes hundreds a move.

In step 4 your search plays the perfect player, half the games as X and half as O. The reference search's results over 20 games:

| simulations per move | draws |
|---|---|
| 25 | 65% |
| 100 | 75% |
| 400 | 90% |
| 800 | 100% |

The numbers wobble a little from run to run, because rollouts are random. The pattern doesn't: more simulations, fewer losses. Your quality metric is the draw rate with 800 simulations a move over 20 games, and 90% or more passes.

## Self-play: the network replaces the rollouts {#self-play}
> A small network learns a policy and a value from the search's own games. The search then needs far fewer simulations, which is the whole idea of AlphaZero.

The last step is a **replay**, a recorded run you watch rather than run in the browser. The recording is **illustrative**: sample data written for the course, labelled as such, to show what a healthy run looks like. You can run the real thing on a laptop CPU in a few minutes.

The loop is:

1. **Self-play.** The search plays against itself. Every position is saved with its visit policy π.
2. **Label.** When the game ends, each saved position gets z, the final result from the view of the player who was to move there: +1, −1 or 0.
3. **Train.** A small network with two outputs learns from those examples. The **policy head** learns to predict π and the **value head** learns to predict z.
4. **Repeat** with the new network inside the search: its policy gives the priors P, and its value replaces the rollout.

Healthy output: losses against the perfect player fall from about 31 in 100 for the untrained net to 0 after around 1,200 self-play games, while the network's errors (policy loss and value loss) go down. With the trained net, 50 simulations a move are enough, where random rollouts needed around 800. Your systems metric is the number of self-play games your run needed before its first clean sweep.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **Board**: a list of 9 cells. `1` is X, `-1` is O, `0` is empty.
- **Legal move**: an empty cell.
- **Line**: one of the 8 rows, columns and diagonals that win.
- **Draw**: a full board with no line. Scored 0.
- **Game tree**: positions as nodes, moves as edges, finished games as leaves.
- **Branching factor**: about how many legal moves a position has.
- **Perfect play**: always choosing a move that's best against a perfect opponent. In tic-tac-toe it always draws.
- **Minimax / negamax**: scoring a tree from the leaves up, each player picking the best for themselves. Negamax flips the sign at each level.
- **MCTS (Monte Carlo tree search)**: building a search tree by repeated simulations and averaging their results.
- **Simulation**: one pass of select, expand, evaluate and back up.
- **Rollout**: random play from a position to the end of the game.
- **Root / node / child / leaf**: the current position, any position in the tree, a position one move further on, and a node with no children yet.
- **Expand**: give a leaf one child per legal move.
- **Prior (P)**: how much the search likes a move before trying it. Uniform in S1, from the policy head in AlphaZero.
- **n / N**: a child's visit count / its parent's visit count.
- **w**: the total of the results backed up through a node.
- **Q**: the average result, `w / n`.
- **PUCT**: the selection score Q + c · P · √N / (1 + n).
- **c_puct**: the weight of the exploration bonus.
- **Exploration / exploitation**: trying moves you know little about / sticking with moves that look best.
- **Backup**: adding a simulation's result to every node on its path, flipping sign at each level.
- **Value**: an estimate of the result from the view of the player to move, from −1 to 1.
- **Policy (π)**: a probability for each move. The visit policy is the root's normalised visit counts.
- **Temperature**: how sharp the policy is. 1 follows the visits, 0 picks the most visited move.
- **Self-play**: the search playing against itself to make training data.
- **Policy head / value head**: the network's two outputs, trained toward π and z.
- **z**: a position's label, the final result from the mover's view.
- **Dirichlet noise**: randomness AlphaZero mixes into the root's priors during self-play, so it keeps trying moves the network dislikes.
- **Draw rate**: the share of games drawn against the perfect player. The lesson's quality metric.
- **Self-play games to converge**: how many self-play games the net needs before it stops losing to perfect play. The systems metric.
- **Replay (illustrative)**: a recorded run you watch in the lesson. Illustrative means sample data, not a real recording.
