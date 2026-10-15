---
paper: Mastering Chess and Shogi by Self-Play (AlphaZero)
gist: S2 reruns the AlphaZero loop on Connect 4. These notes cover the paper's choices that change, or matter more, on a game you can't search to the end.
---

## Same loop, smaller game {#loop}
> Self-play with search, train the network on the search's output, repeat: S2 keeps every piece of that loop and shrinks the scale.

S1's notes cover the algorithm itself. On Connect 4 the loop is the same: MCTS guided by a policy/value network plays both sides, each position is stored with its visit distribution π, the game's result z labels it, and the network trains on (state, π, z). The paper ran this with thousands of TPUs making games. S2's replay runs it on one GPU and a few CPU workers, which is enough because Connect 4 is far smaller than chess.

## Draws and the value target {#draws}
> AlphaZero's value predicts the expected result, not a win probability, so a draw is a clean 0.

Connect 4 can end with a full board and no four, just as chess can end drawn. Following the paper, `to_examples` labels a drawn game 0 for both sides, and wins and losses ±1 from the view of the player to move. A value head trained this way predicts the average result from a position, which is exactly what backup adds up in the tree.

## Exploration in self-play {#exploration}
> Noise on the root priors and sampled early moves keep self-play from replaying the same few games.

The paper adds Dirichlet noise to the root's priors during self-play and picks moves from the visit counts, so the network keeps seeing new positions. S2's `self_play_game` samples its first 8 moves from the visits and then plays the most visited one; the replay's run also adds root noise. Without some randomness, a deterministic search plays the same opening every time, and the training data repeats.

## Symmetry: one shortcut S2 can take back {#symmetry}
> AlphaZero gave up board symmetries because chess has none, but Connect 4 is left-right symmetric, so mirroring doubles the data for free.

AlphaGo Zero augmented Go data with rotations and reflections. AlphaZero dropped that so the same algorithm would work on chess and shogi. Connect 4 does have a mirror symmetry: flip the board left to right and flip π with it, and you get another correct example. The replay's run mirrors every position. This is a game-specific trick, which is what AlphaZero set out to avoid, so it's a choice you make knowingly.

## Measuring progress {#evaluation}
> The paper rated its networks against strong engines; S2 uses a fixed depth-4 minimax bot as its ruler.

The paper tracked strength with Elo ratings and matches against Stockfish, Elmo and AlphaGo Zero. S2 needs something cheaper that never changes between generations: the negamax bot from step 4. Its strength is fixed, so a rising win rate against it means the agent improved, not that the opponent got weaker. The bot can only see 4 moves ahead, which is the gap a trained value head learns to close.
