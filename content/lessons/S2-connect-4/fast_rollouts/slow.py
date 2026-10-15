

import copy


def any_win(board):
    """Check every cell for four in a row. Returns the winner, or 0."""
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] != 0 and is_win(board, r, c):
                return board[r][c]
    return 0


def rollout_copy(board, player, rng):
    """The slow way: copy the board, then check all of it after every move."""
    board = copy.deepcopy(board)
    turn = player
    while any(cell == 0 for cell in board[0]):
        drop(board, rng.choice(legal_moves(board)), turn)
        won = any_win(board)
        if won:
            return won * player
        turn = -turn
    return 0
