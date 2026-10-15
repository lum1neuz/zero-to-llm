import random
from harness import test, fail, warn, learner_fn, LearnerError, script_error, stdout

ROWS, COLS = 6, 7


def board_of(*rows):
    rows = ["......."] * (ROWS - len(rows)) + list(rows)
    return [[{".": 0, "X": 1, "O": -1}[ch] for ch in row] for row in rows]


def pieces(board):
    return sum(1 for row in board for v in row if v != 0)


def snapshot(board):
    return [row[:] for row in board]


# Forced endings, top row first. Rollouts have no choice to make in any of them.
# One free cell, (0, 4): O fills it and completes the top row.
O_WINS = board_of("OXOO.OO", "XOXOXXX", "OXOXOXO", "XOOXOXO", "XOXXXOX", "XXXOXOO")
# One free cell, (0, 1): whoever fills it, nobody has four. A draw.
DRAW = board_of("X.OXXOX", "OXXOOXX", "OOXXXOO", "XXXOOOX", "OOOXXXO", "OOXOXXO")
# Two free cells in column 6: X fills (1, 6) without a four, then O fills (0, 6) and wins.
X_LOSES = board_of("XOXOOO.", "XOOOXX.", "OXOXOOX", "OXOXXXO", "XOXXXOO", "OXXOXOO")


def midgame():
    """A position 12 moves in with no four on it yet."""
    rng = random.Random(11)
    while True:
        b, player, ok = [[0] * COLS for _ in range(ROWS)], 1, True
        for _ in range(12):
            col = rng.choice([c for c in range(COLS) if b[0][c] == 0])
            row = max(r for r in range(ROWS) if b[r][col] == 0)
            b[row][col] = player
            player = -player
        if not any(four_through(b, r, c) for r in range(ROWS) for c in range(COLS) if b[r][c]):
            return b


def four_through(b, row, col):
    p = b[row][col]
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        n = 1
        for s in (1, -1):
            r, c = row + s * dr, col + s * dc
            while 0 <= r < ROWS and 0 <= c < COLS and b[r][c] == p:
                n, r, c = n + 1, r + s * dr, c + s * dc
        if n >= 4:
            return True
    return False


class WatchedRow(list):
    """A board row that notes when anything writes to it."""
    writes = 0

    def __setitem__(self, i, v):
        WatchedRow.writes += 1
        super().__setitem__(i, v)


def call_rollout(L, board, player, seed=3):
    rollout = learner_fn(L, "rollout")
    try:
        return rollout(board, player, random.Random(seed))
    except LearnerError as e:
        if "IndexError" in str(e) and ("empty" in str(e) or "range" in str(e)):
            fail("`rollout` crashed when the board filled up with nobody winning: `rng.choice` got an empty list of moves. "
                 "Stop when no legal move is left. That game is a draw, 0.")
        raise


@test("undo_top_piece", "undo takes the top piece back out of a column")
def _(L):
    undo = learner_fn(L, "undo")
    b = board_of("..X....", "..O....", "..X....")
    undo(b, 2)
    col = [b[r][2] for r in range(ROWS)]
    if col == [0, 0, 0, 0, -1, 1]:
        return
    if col == [0, 0, 0, 1, -1, 1]:
        fail("After `undo(board, 2)` column 2 still has all 3 pieces. Is `undo` still the `...` placeholder?")
    if col == [0, 0, 0, 1, -1, 0]:
        fail("`undo` took out the *bottom* piece of column 2, the first one dropped. drop fills a column from the bottom up, "
             "so the piece to take back is the highest one.")
    if col == [0] * ROWS:
        fail("`undo` emptied all of column 2. It should take out one piece, the top one: stop after the first piece you clear.")
    fail(f"Column 2 held X, O, X from the bottom up. After `undo(board, 2)` it should hold X, O. Top to bottom it now reads {col}.")


@test("board_unchanged", "rollout leaves the board exactly as it found it")
def _(L):
    b = midgame()
    before = snapshot(b)
    result = call_rollout(L, b, 1)
    if result is None or result is Ellipsis:
        fail("`rollout` returns nothing. Is it still the `...` placeholder?")
    if b != before:
        extra = pieces(b) - pieces(before)
        if extra > 0:
            fail(f"`rollout` left {extra} new pieces on the board. run_mcts reuses this one board for every simulation, "
                 "so they'd pile up from one simulation to the next. Undo each move you made, newest first.")
        fail("After `rollout` the board isn't the one it was given. Every move it plays has to be taken back out.")
    WatchedRow.writes = 0
    watched = [WatchedRow(row) for row in before]
    call_rollout(L, watched, 1)
    if WatchedRow.writes == 0:
        warn("note: rollout never wrote to the board it was given, so it plays on a copy. That works, but copying is "
             "half of what makes rollout_copy slow. Compare the two sims/s lines.")


@test("result_for_player", "rollout returns the result for the player who starts")
def _(L):
    got = call_rollout(L, snapshot(O_WINS), -1)
    if got != 1:
        if got == -1:
            fail("O starts this rollout and wins with the last free cell, but `rollout` returns −1. That's *who* won. "
                 "MCTS needs the result for the player it passed in: +1 means they won.")
        fail(f"O starts this rollout and wins with the last free cell, so the result for O is 1. `rollout` returns {got!r}.")
    got = call_rollout(L, snapshot(X_LOSES), 1)
    if got != -1:
        fail(f"X starts this rollout and has to play into column 6; then O drops on top and wins. The result for X is −1. "
             f"`rollout` returns {got!r}.")
    got = call_rollout(L, snapshot(DRAW), 1)
    if got != 0:
        fail(f"X fills the last free cell and nobody has four: a draw, 0. `rollout` returns {got!r}.")


@test("script_runs", "The speed test runs both rollouts")
def _(L):
    err = script_error()
    if err and "is full" in err["diagnosis"]:
        fail("Your script stopped: a column filled up during the speed test. run_mcts takes its moves back out with your "
             "`undo`, so the board only stays clean once `undo` works.")
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if stdout().count("sims/s") < 2:
        fail("The speed test at the bottom didn't print both lines. Put the loop over `(rollout_copy, rollout)` back.")
