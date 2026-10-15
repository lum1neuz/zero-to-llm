import math
import random
from harness import test, fail, learner_fn, LearnerError, script_error

ROWS, COLS = 6, 7


def board_of(*rows):
    rows = ["......."] * (ROWS - len(rows)) + list(rows)
    return [[{".": 0, "X": 1, "O": -1}[ch] for ch in row] for row in rows]


def txt(board):
    return "`" + " / ".join("".join(".XO"[v] for v in row) for row in board) + "`"


# X has three along the bottom, columns 2 to 4. Column 1 is O's, so the only open end is column 5.
PUZZLE = board_of("...O...", ".OXXX..")

# ---- a reference search, written independently of the learner's helpers ----
WINDOWS = [[(r + i * dr, c + i * dc) for i in range(4)]
           for r in range(ROWS) for c in range(COLS) for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
           if 0 <= r + 3 * dr < ROWS and 0 <= c + 3 * dc < COLS]
POINTS = [0, 1, 4, 16, 0]


def ref_heuristic(b, p):
    score = 0
    for cells in WINDOWS:
        vals = [b[r][c] for r, c in cells]
        mine, theirs = vals.count(p), vals.count(-p)
        if not theirs:
            score += POINTS[mine]
        elif not mine:
            score -= POINTS[theirs]
    return score


def ref_four(b, row, col):
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


def ref_negamax(b, depth, p, negate=True, swap=True):
    """negate=False or swap=False give the two classic slips, so the tests can name them."""
    moves = [c for c in range(COLS) if b[0][c] == 0]
    if depth == 0 or not moves:
        return ref_heuristic(b, p), None
    best = (-math.inf, None)
    for col in moves:
        row = max(r for r in range(ROWS) if b[r][col] == 0)
        b[row][col] = p
        if ref_four(b, row, col):
            score = 1000 + depth
        else:
            reply = ref_negamax(b, depth - 1, -p if swap else p, negate, swap)[0]
            score = -reply if negate else reply
        b[row][col] = 0
        if score > best[0]:
            best = (score, col)
    return best


def slip(board, depth, player, got):
    """Name the slip when the learner's result matches a known-wrong search, else None."""
    if tuple(got) == ref_negamax([row[:] for row in board], depth, player, swap=False):
        return ("The recursive call plays the same player again, so the search imagines one side making every move. "
                "After player drops a piece, whose turn is it?")
    if tuple(got) == ref_negamax([row[:] for row in board], depth, player, negate=False):
        return ("The score one level deeper is from the opponent's view, and you add it as it is. "
                "The opponent's best result is your worst: negate it.")
    return None


def search(L, board, depth, player):
    negamax = learner_fn(L, "negamax")
    try:
        out = negamax(board, depth, player)
    except LearnerError as e:
        if "is full" in str(e):
            fail("`negamax` filled a column to the top and crashed. It drops a piece for every move it tries, "
                 "but never takes it back out. Undo each move after you've scored it.")
        raise
    if not (isinstance(out, tuple) and len(out) == 2):
        fail("`negamax` should return `(score, col)`. Keep the `return best_score, best_col` line at the end.")
    return out


@test("board_restored", "negamax leaves the board as it found it")
def _(L):
    b = [row[:] for row in PUZZLE]
    search(L, b, 2, 1)
    if b != PUZZLE:
        extra = sum(v != 0 for row in b for v in row) - sum(v != 0 for row in PUZZLE for v in row)
        fail(f"After a depth-2 search the board has {extra} extra pieces on it. "
             "Every piece negamax drops to try a move has to come back out before it tries the next one.")


@test("takes_the_win", "Takes a win that's one move away")
def _(L):
    for depth in (1, 3):
        score, col = search(L, [row[:] for row in PUZZLE], depth, 1)
        if col != 5:
            fail(f"X has three along the bottom and can win right now in column 5, but at depth {depth} negamax picks column {col}. "
                 "After dropping a piece, does your code check whether that piece just won?")
        if score != 1000 + depth:
            fail(f"X wins at once in column 5, which should score 1000 + depth = {1000 + depth} at depth {depth}. "
                 f"Your score is {score}.")


@test("blocks_the_threat", "Blocks the opponent's four")
def _(L):
    for depth in (2, 4):
        got = search(L, [row[:] for row in PUZZLE], depth, -1)
        if got[1] != 5:
            story = slip(PUZZLE, depth, -1, got) or ("The score one level deeper is from X's view. "
                                                    "Is the opponent's best result good or bad for O?")
            fail(f"O to move must block column 5, or X completes four there next turn. "
                 f"At depth {depth} negamax picks column {got[1]}. {story}")


def positions():
    rng = random.Random(9)
    out = []
    while len(out) < 6:
        b, p, ok = [[0] * COLS for _ in range(ROWS)], 1, True
        for _ in range(rng.randint(6, 16)):
            col = rng.choice([c for c in range(COLS) if b[0][c] == 0])
            row = max(r for r in range(ROWS) if b[r][col] == 0)
            b[row][col] = p
            if ref_four(b, row, col):
                ok = False
                break
            p = -p
        if ok:
            out.append((b, p))
    return out


@test("matches_reference", "Matches a reference search on 6 positions")
def _(L):
    for b, p in positions():
        want = ref_negamax([row[:] for row in b], 3, p)
        got = search(L, [row[:] for row in b], 3, p)
        if tuple(got) != want:
            who = "X" if p == 1 else "O"
            story = slip(b, 3, p, got) or ("Check each piece: the win score is 1000 + depth, the recursive call is the "
                                           "*other* player's turn one level shallower, and its score is negated.")
            fail(f"On {txt(b)} with {who} to move, a reference depth-3 negamax gives (score, col) = {want}, and yours gives "
                 f"{tuple(got)}. {story}")


@test("script_runs", "The puzzle runs")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
