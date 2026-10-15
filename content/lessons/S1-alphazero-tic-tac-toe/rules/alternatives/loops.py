import math
import random

# A board is a list of 9 cells, left to right, top to bottom:
#    0 | 1 | 2
#    3 | 4 | 5
#    6 | 7 | 8
# 1 is X, -1 is O, 0 is empty. X moves first.

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def legal_moves(board):
    # TODO: the indices of the empty cells, smallest first
    moves = []
    for i in range(9):
        if board[i] == 0:
            moves.append(i)
    return moves


def winner(board):
    # TODO: 1 or -1 if that player owns all three cells of a line in LINES,
    # 0 if the board is full and nobody does (a draw), None while the game goes on
    for line in LINES:
        cells = {board[i] for i in line}
        if len(cells) == 1 and 0 not in cells:
            return cells.pop()
    if 0 in board:
        return None
    return 0


board = [1, -1, 1,
         0, -1, 0,
         0, 1, 0]
print("legal moves:", legal_moves(board))
print("winner:", winner(board))
