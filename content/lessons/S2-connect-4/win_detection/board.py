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
