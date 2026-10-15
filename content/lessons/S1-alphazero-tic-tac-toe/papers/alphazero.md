---
paper: Mastering Chess and Shogi by Self-Play (AlphaZero)
gist: One algorithm reaches superhuman chess, shogi and Go from the rules alone, by pairing tree search with a single network trained only on its own games.
---

## The claim {#claim}
> Given nothing but the rules, the same algorithm with the same settings learns chess, shogi and Go well enough to beat the strongest programs.

Chess engines before AlphaZero were built from decades of human knowledge: hand-tuned evaluation functions, opening books, endgame tables and many search tricks specific to the game. AlphaGo Zero had already learned Go with no human games. AlphaZero asks whether that recipe generalises. It drops the Go-specific parts and runs one algorithm on three very different games. The only game-specific input is the rules, which tell it the legal moves and when a game is over.

## One network, two heads {#network}
> A single network takes a position and returns both a policy over moves and a value that predicts the final result.

The policy p is a probability for each legal move: where a strong player would probably play. The value v is a number between −1 and 1: the expected outcome for the player to move. Both come out of one shared body (a deep residual network), so one call answers both questions. In S1 you replace both with something simple: a uniform policy and a random game played to the end.

## Search guided by the network {#search}
> MCTS uses the policy as a prior and the value in place of rollouts, and the root's visit counts become the move choice.

Each simulation walks down the tree, picking at every node the child that balances a high average value Q against a bonus that favours moves with a high prior and few visits so far. That's the PUCT rule you write in step 2. At a new leaf the network is called once: its policy becomes the priors of the leaf's children and its value is backed up the path, with the sign switching between the two players at each level. There are no random rollouts at all. After the simulations, the visit counts at the root say which moves the search trusted, and the move is picked from them.

## Learning from self-play {#self-play}
> Every self-play position becomes a training example: the search's visit distribution is the policy target, and the game's final result is the value target.

The network plays itself, using the search at every move. Each position is stored with π, the normalised visit counts, and later labelled with z, the final result from the view of the player to move there. Training pushes the policy head toward π (cross-entropy) and the value head toward z (squared error), with weight decay on top. The search is stronger than the raw network, so training on its output makes the network better, which makes the next round of search better again. Root priors get Dirichlet noise during self-play, so the search sometimes tries moves the network dislikes.

## What changed from AlphaGo Zero {#changes}
> The Go-specific shortcuts are gone: no board symmetries, results can be draws, and one network keeps training instead of being replaced after evaluation matches.

AlphaGo Zero used Go's rotations and reflections to augment its data and its search. Chess and shogi aren't symmetric that way, so AlphaZero doesn't. Go has no draws, so AlphaGo Zero's value estimated a win probability. AlphaZero's value estimates the expected result, so a draw counts as 0. AlphaGo Zero also kept a "best player" and only replaced it when a new network beat it in a match. AlphaZero just keeps updating one network and always self-plays with the latest version. Hyperparameters are shared across games; only the scale of the exploration noise depends on how many legal moves a game typically has.

## Results {#results}
> Within hours of self-play training it beat Stockfish at chess, Elmo at shogi and, at Go, the AlphaGo version that beat Lee Sedol, while searching far fewer positions per second.

Training used thousands of TPUs to generate self-play games, which is what made a few hours of wall-clock time possible. In a 100-game chess match it didn't lose a game against Stockfish. It also looked at roughly a thousandth as many positions per second as Stockfish did, so the strength comes from a much better sense of which positions are worth looking at and how good they are, not from searching more.

## How the course uses it {#course}
> S1 builds the search loop by hand on tic-tac-toe; the optional replay and S2 add the network and self-play training.

The four challenges map onto the search: legal moves and results (the rules AlphaZero is given), PUCT selection, the sign-flipping backup, and visit counts into a policy. Random rollouts stand in for the value head and a uniform prior stands in for the policy head, which is why S1's search needs hundreds of simulations a move where the replay's trained net needs 50. S2 keeps the same loop and adds the parts that matter on a bigger game: speed, self-play data in the (state, π, z) form above, and a fixed opponent to measure progress.
