def legal_moves(board):
    # TODO: the indices of the empty cells, smallest first
    return ...


def winner(board):
    # TODO: 1 or -1 if that player owns all three cells of a line in LINES,
    # 0 if the board is full and nobody does (a draw), None while the game goes on
    return ...


board = [1, -1, 1,
         0, -1, 0,
         0, 1, 0]
print("legal moves:", legal_moves(board))
print("winner:", winner(board))
