import random
from harness import test, fail, learner_fn, LearnerError

ROWS, COLS = 6, 7
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def board_of(*rows):
    """Rows top to bottom as strings like "XX.O..."; missing rows at the top are empty."""
    rows = ["......."] * (ROWS - len(rows)) + list(rows)
    return [[{".": 0, "X": 1, "O": -1}[ch] for ch in row] for row in rows]


def row_txt(row):
    return "`" + " ".join(".XO"[v] for v in row) + "`"


def board_txt(board):
    return "`" + " / ".join("".join(".XO"[v] for v in row) for row in board) + "`"


def ref_run(board, row, col, dr, dc):
    player, count = board[row][col], 0
    r, c = row + dr, col + dc
    while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == player:
        count += 1
        r, c = r + dr, c + dc
    return count


def any_piece_run(board, row, col, dr, dc):
    count, r, c = 0, row + dr, col + dc
    while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] != 0:
        count, r, c = count + 1, r + dr, c + dc
    return count


def no_stop_run(board, row, col, dr, dc):
    count, r, c = 0, row + dr, col + dc
    while 0 <= r < ROWS and 0 <= c < COLS:
        count += board[r][c] == board[row][col]
        r, c = r + dr, c + dc
    return count


def ref_win(board, row, col):
    return any(1 + ref_run(board, row, col, dr, dc) + ref_run(board, row, col, -dr, -dc) >= 4 for dr, dc in DIRS)


def run(L, board, row, col, dr, dc):
    fn = learner_fn(L, "run_length")
    try:
        got = fn([r[:] for r in board], row, col, dr, dc)
    except LearnerError as e:
        if "IndexError" in str(e):
            way = {(0, 1): "right", (0, -1): "left", (1, 0): "down", (-1, 0): "up"}.get((dr, dc), f"along ({dr}, {dc})")
            fail(f"`run_length` walked off the board: going {way} from ({row}, {col}) it tried to read a cell that doesn't exist, "
                 "and Python raised IndexError. Check that the next cell is on the board before you read it.")
        raise
    if got is Ellipsis:
        fail("`count` is still `...`, the placeholder.")
    if got is None:
        fail("`run_length` returns nothing. Keep the `return count` line at the end.")
    return got


def win(L, board, row, col):
    fn = learner_fn(L, "is_win")
    got = fn([r[:] for r in board], row, col)
    if got is Ellipsis:
        fail("`is_win` still returns `...`, the placeholder.")
    if got is None:
        fail("`is_win` returns nothing. It has to return True or False.")
    return bool(got)


# (board, row, col, dr, dc). None of these runs reaches an edge.
C1 = (board_of("XXXOOX."), 5, 0, 0, 1)
C2 = (board_of("...X...", "...X...", "...O..."), 3, 3, 1, 0)
C3 = (board_of("..X....", "...X...", "....O.."), 3, 2, 1, 1)


@test("run_length_counts", "run_length counts the same player's pieces in one direction")
def _(L):
    got = [run(L, *case) for case in (C1, C2, C3)]
    want = [ref_run(*case) for case in (C1, C2, C3)]
    if got == want:
        return
    row = row_txt(C1[0][5])
    start = f"On a bottom row of {row}, going right from the X at column 0, the run is 2: the X's at columns 1 and 2."
    if got == [w + 1 for w in want]:
        fail(f"{start} You got {got[0]}. Does your count include (row, col) itself? `is_win` adds the new piece once on its own, "
             "so `run_length` counts only the neighbours.")
    if got == [any_piece_run(*case) for case in (C1, C2, C3)]:
        fail(f"{start} You got 5: you counted O's pieces as well. A run only counts cells holding the *same* player as (row, col).")
    if got == [no_stop_run(*case) for case in (C1, C2, C3)]:
        fail(f"{start} You got 3: you kept going past the O at column 3 and counted the X at column 5 too. "
             "A run stops at the first cell that isn't the player's.")
    if got[0] != want[0]:
        fail(f"{start} You got {got[0]}.")
    fail(f"Down from the X at (3, 3), over an X then an O, the run is 1, and down-right from the X at (3, 2), "
         f"over an X then an O, it's 1 too. You got {got[1]} and {got[2]}.")


@test("stays_on_board", "run_length stops at the board's edges")
def _(L):
    b = board_of("X....XX")
    run(L, b, 5, 6, 0, 1)
    run(L, b, 5, 0, 1, 0)
    got = run(L, b, 5, 0, 0, -1)
    if got != 0:
        fail(f"On a bottom row of {row_txt(b[5])}, going left from column 0, there's no column −1, so the run is 0. You got {got}: "
             "in Python `board[5][-1]` is the *last* column, so your run wrapped round to columns 6 and 5. "
             "Check that the next row and column are 0 or more, not only that they're under ROWS and COLS.")
    b = board_of("...X...", ".......", ".......", ".......", "...X...", "...X...")
    got = run(L, b, 0, 3, -1, 0)
    if got != 0:
        fail(f"Going up from (0, 3), the top row, the run is 0. You got {got}: `board[-1]` is the *bottom* row in Python, "
             "so your run wrapped round. Check that the next row is 0 or more.")


@test("win_through_middle", "is_win spots a four when the new piece lands in the middle")
def _(L):
    b = board_of("XXXX...")
    if win(L, b, 5, 2):
        return
    left, right = win(L, b, 5, 0), win(L, b, 5, 3)
    if left and not right:
        fail("`X X X X` along the bottom: is_win sees it when the new piece is at the left end, but not at column 3 or 2. "
             "You count from the new piece in one direction only. Count both ways along each line, (dr, dc) and (−dr, −dc).")
    if right and not left:
        fail("`X X X X` along the bottom: is_win sees it when the new piece is at the right end, but not at column 0 or 2. "
             "You count from the new piece in one direction only. Count both ways along each line, (dr, dc) and (−dr, −dc).")
    if not left and not right:
        fail("`X X X X` along the bottom, and is_win says False wherever the new piece is. Three neighbours plus the new piece "
             "make four: does your total count the new piece itself?")
    fail("`X X X X` along the bottom, with the new piece at column 2: is_win says False.")


@test("all_directions", "is_win checks across, down and both diagonals")
def _(L):
    cases = [
        (board_of("X......", "X......", "X......", "X......"), 2, 0, "straight down column 0, from (2, 0) to (5, 0)"),
        (board_of("X......", ".X.....", "..X....", "...X..."), 2, 0, "on the diagonal falling to the right, from (2, 0) to (5, 3)"),
        (board_of("...X...", "..X....", ".X.....", "X......"), 2, 3, "on the diagonal rising to the right, from (5, 0) up to (2, 3)"),
    ]
    for b, row, col, where in cases:
        if not win(L, b, row, col):
            fail(f"X has four {where}, and the new piece is at ({row}, {col}). is_win says False. "
                 "Which of your directions runs along that line?")


@test("exactly_four", "Three in a row isn't a win; five is")
def _(L):
    if win(L, board_of("XXX...."), 5, 2):
        fail("`X X X` along the bottom row is only three, but is_win says True. Count again: what's your total for this line, "
             "and what does it have to reach?")
    if not win(L, board_of("XXXXX.."), 5, 2):
        fail("Filling the gap in `X X _ X X` makes five in a row, and five in a row contains a four. is_win says False. "
             "What does your comparison do with a total of 5?")


@test("matches_scan", "Agrees with a full-board check on 200 random games")
def _(L):
    fn = learner_fn(L, "is_win")
    rng = random.Random(4)
    for _ in range(200):
        b = [[0] * COLS for _ in range(ROWS)]
        player = 1
        while True:
            open_cols = [c for c in range(COLS) if b[0][c] == 0]
            if not open_cols:
                break
            col = rng.choice(open_cols)
            row = max(r for r in range(ROWS) if b[r][col] == 0)
            b[row][col] = player
            want = ref_win(b, row, col)
            got = fn([r[:] for r in b], row, col)
            if got is Ellipsis or got is None:
                fail("`is_win` returns nothing yet. Is it still the `...` placeholder?")
            if bool(got) != want:
                who = "X" if player == 1 else "O"
                fail(f"{who} dropped into column {col} and landed at ({row}, {col}). Board, top row first: {board_txt(b)}. "
                     f"is_win says {bool(got)}, but it's {want}.")
            if want:
                break
            player = -player
