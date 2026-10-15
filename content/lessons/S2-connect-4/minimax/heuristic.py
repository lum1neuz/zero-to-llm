

# every line of four cells on the board: 24 across, 21 down, 12 on each diagonal
WINDOWS = []
for r in range(ROWS):
    for c in range(COLS):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            cells = [(r + i * dr, c + i * dc) for i in range(4)]
            if all(0 <= rr < ROWS and 0 <= cc < COLS for rr, cc in cells):
                WINDOWS.append(cells)

POINTS = [0, 1, 4, 16, 0]    # for 0, 1, 2 or 3 pieces in a window nobody else is in (4 is a win, which is_win finds)


def heuristic(board, player):
    """Windows only player is in, minus windows only the opponent is in, weighted by how full they are."""
    score = 0
    for cells in WINDOWS:
        values = [board[r][c] for r, c in cells]
        mine, theirs = values.count(player), values.count(-player)
        if not theirs:
            score += POINTS[mine]
        elif not mine:
            score -= POINTS[theirs]
    return score
