

def visit_policy(root, temperature=1.0):
    # TODO: 9 probabilities, one per cell, from the visit counts n of the root's children.
    # A cell with no child gets 0. Each move's share is proportional to n ** (1 / temperature):
    # temperature 1 follows the visits, lower is sharper, and temperature 0 puts all of it
    # on the most visited move.
    ...


def play_perfect(mcts_player, sims, rng):
    """One game: your MCTS plays mcts_player, the perfect player plays the other side."""
    board = [0] * 9
    player = 1
    while winner(board) is None:
        if player == mcts_player:
            pi = visit_policy(run_mcts(board, player, sims, rng), temperature=0)
            move = pi.index(max(pi))
        else:
            move = perfect_move(board, player)
        board[move] = player
        player = -player
    return winner(board)


SIMS = 200
GAMES = 10
rng = random.Random(0)
pi = visit_policy(run_mcts([0] * 9, 1, SIMS, rng))
print("Policy target from the empty board:", [round(p, 2) for p in pi])
results = [play_perfect(1 if g % 2 == 0 else -1, SIMS, rng) for g in range(GAMES)]
draws = results.count(0)
print(f"{SIMS} sims: {draws}/{GAMES} draws vs perfect play ({100 * draws / GAMES:.0f}%)")
