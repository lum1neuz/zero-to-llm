

def snapshot(board, player):
    # TODO: a copy of board as player sees it: player's own pieces 1, the opponent's -1.
    # A real copy, so the moves played after it can't change it.
    return ...


def to_examples(history, result):
    # history holds one (state, player, pi) per move. result: 1 if X won, -1 if O won, 0 a draw.
    # TODO: one (state, pi, z) per move, where z is the final result from the view of
    # the player who was to move in that state
    return ...


def self_play_game(n_sims, rng, explore_moves=8):
    """One game of the search against itself. Returns its training examples."""
    board = new_board()
    player = 1
    history = []
    while True:
        root = run_mcts(board, player, n_sims, rng)
        # sample from the visits early on for varied openings, then play the most visited move
        pi = visit_policy(root, 1.0 if len(history) < explore_moves else 0)
        history.append((snapshot(board, player), player, pi))
        col = rng.choices(range(COLS), weights=pi)[0]
        row = drop(board, col, player)
        if is_win(board, row, col):
            return to_examples(history, player)
        if not legal_moves(board):
            return to_examples(history, 0)
        player = -player


examples = self_play_game(30, random.Random(1))
state, pi, z = examples[-1]
print(f"{len(examples)} examples. The last mover's z: {z}, the move before: {examples[-2][2]}")
print("The last state, from the last mover's side, so their pieces show as X:")
show(state)
