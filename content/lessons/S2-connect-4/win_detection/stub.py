

def run_length(board, row, col, dr, dc):
    # TODO: step away from (row, col) along (dr, dc), one cell at a time, and count the cells
    # that hold the same player as (row, col). Stop at the edge of the board or at any other
    # cell. Don't count (row, col) itself.
    count = ...
    return count


def is_win(board, row, col):
    # TODO: True if the piece at (row, col) is part of four or more in a row: across (0, 1),
    # down (1, 0), or along either diagonal, (1, 1) and (1, -1). Count both ways along each.
    return ...


board = new_board()
for col, player in [(0, 1), (0, -1), (1, 1), (1, -1), (3, 1), (3, -1)]:
    drop(board, col, player)
row = drop(board, 2, 1)
show(board)
print("X drops into column 2. Four in a row?", is_win(board, row, 2))
