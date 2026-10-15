import math
import random
import time

ROWS, COLS = 6, 7


def new_board():
    return [[0] * COLS for _ in range(ROWS)]


def legal_moves(board):
    # a column is open while its top cell is empty
    return [c for c in range(COLS) if board[0][c] == 0]


def drop(board, col, player):
    """Drop player's piece into col. Returns the row it lands in."""
    for row in range(ROWS - 1, -1, -1):
        if board[row][col] == 0:
            board[row][col] = player
            return row
    raise ValueError(f"column {col} is full")


def show(board):
    print("\n".join(" ".join(".XO"[v] for v in row) for row in board))


def run_length(board, row, col, dr, dc):
    # TODO: step away from (row, col) along (dr, dc), one cell at a time, and count the cells
    # that hold the same player as (row, col). Stop at the edge of the board or at any other
    # cell. Don't count (row, col) itself.
    player = board[row][col]
    count = 0
    r, c = row, col
    while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == player:
        count += 1
        r, c = r + dr, c + dc
    return count


def is_win(board, row, col):
    # TODO: True if the piece at (row, col) is part of four or more in a row: across (0, 1),
    # down (1, 0), or along either diagonal, (1, 1) and (1, -1). Count both ways along each.
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if 1 + run_length(board, row, col, dr, dc) + run_length(board, row, col, -dr, -dc) >= 4:
            return True
    return False


board = new_board()
for col, player in [(0, 1), (0, -1), (1, 1), (1, -1), (3, 1), (3, -1)]:
    drop(board, col, player)
row = drop(board, 2, 1)
show(board)
print("X drops into column 2. Four in a row?", is_win(board, row, 2))
