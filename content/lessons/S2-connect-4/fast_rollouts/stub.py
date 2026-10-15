

def undo(board, col):
    # TODO: take the top piece back out of column col, the reverse of drop
    ...


def rollout(board, player, rng):
    # TODO: play random legal moves on board itself, player first, until someone wins or the
    # board is full. Then undo every move you made, newest first, so board is exactly as you
    # found it. Return the result for player: 1 a win, -1 a loss, 0 a draw.
    ...


rng = random.Random(0)
for fn in (rollout_copy, rollout):
    start = time.perf_counter()
    root = run_mcts(new_board(), 1, 300, rng, rollout_fn=fn)
    rate = 300 / (time.perf_counter() - start)
    print(f"{fn.__name__:<13} {rate:6.0f} sims/s")
