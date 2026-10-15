

def negamax(board, depth, player):
    """(score, col): the best column for player, looking depth moves ahead. Scores are from
    player's view. A win scores 1000 + depth, so a sooner win scores higher."""
    moves = legal_moves(board)
    if depth == 0 or not moves:
        return heuristic(board, player), None
    best_score, best_col = -math.inf, None
    for col in moves:
        # TODO: drop player's piece into col. It scores 1000 + depth if it wins on the spot;
        # otherwise minus the opponent's best score one level deeper. Then take it back out.
        score = ...
        if score > best_score:
            best_score, best_col = score, col
    return best_score, best_col


def minimax_bot(board, player):
    return negamax(board, 4, player)[1]


puzzle = new_board()
for col, player in [(2, 1), (1, -1), (3, 1), (3, -1), (4, 1)]:
    drop(puzzle, col, player)
show(puzzle)
print("If X were to move, depth 1:", negamax(puzzle, 1, 1))
print("O to move, the depth-4 bot plays column", minimax_bot(puzzle, -1))
