from harness import test, fail, learner_fn

X, O = 1, -1
REF_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

EMPTY = [0] * 9
# Unfinished, and every line already has a piece in it, so only a real "is it full?" check says None.
SPREAD = [X, 0, 0,
          0, O, 0,
          0, 0, X]
# Full, and nobody owns a line.
DRAW = [X, O, X,
        X, O, O,
        O, X, X]
# Full, and X's last move completed the 0-4-8 diagonal.
FULL_WIN = [X, O, X,
            O, X, O,
            O, X, X]
# (board, winner, what wins). W2's top row is full but mixed, and comes first in LINES.
WINS = [
    ([X, X, X, O, O, 0, 0, 0, 0], X, "the top row"),
    ([O, O, O, X, X, 0, X, 0, 0], O, "the top row"),
    ([X, O, 0, X, O, 0, X, 0, 0], X, "the left column"),
    ([X, O, X, 0, O, X, 0, O, 0], O, "the middle column"),
    ([X, X, O, X, O, 0, O, 0, 0], O, "the 2-4-6 diagonal"),
]


def show(board):
    marks = {1: "X", -1: "O", 0: "."}
    return " / ".join(" ".join(marks.get(v, "?") for v in board[i:i + 3]) for i in (0, 3, 6))


def name(v):
    return {1: "X (1)", -1: "O (−1)"}.get(v, repr(v))


def ref_moves(board):
    return [i for i in range(9) if board[i] == 0]


def ref_winner(board):
    for a, b, c in REF_LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0 if 0 not in board else None


def reachable():
    """Every position that can come up in a real game, 5,478 of them."""
    seen, stack = {tuple(EMPTY)}, [(EMPTY, X)]
    while stack:
        board, player = stack.pop()
        if ref_winner(board) is not None:
            continue
        for m in ref_moves(board):
            nxt = board[:]
            nxt[m] = player
            if tuple(nxt) not in seen:
                seen.add(tuple(nxt))
                stack.append((nxt, -player))
    return [list(b) for b in sorted(seen)]


LINE_NAMES = {(0, 1, 2): "top row", (3, 4, 5): "middle row", (6, 7, 8): "bottom row", (0, 3, 6): "left column",
              (1, 4, 7): "middle column", (2, 5, 8): "right column", (0, 4, 8): "0-4-8 diagonal", (2, 4, 6): "2-4-6 diagonal"}


def mixed_full_line(board, got):
    """The message for winner() calling a full but mixed line a win, or None if that's not what happened."""
    for a, b, c in REF_LINES:
        cells = (board[a], board[b], board[c])
        if 0 not in cells and len(set(cells)) > 1:
            if same(got, board[a]):
                marks = " ".join({1: "X", -1: "O"}[v] for v in cells)
                return (f"On `{show(board)}` `winner` says {name(got)} won. The {LINE_NAMES[(a, b, c)]} is {marks}: every cell taken, "
                        "but not by one player. What do the three cells of a winning line have in common?")
            return None
    return None


def same(got, want):
    """winner results compare by type too: True isn't 1 and False isn't 0 here."""
    return type(got) is type(want) and got == want


def call_winner(L, board):
    win = learner_fn(L, "winner")
    got = win(list(board))
    if got is Ellipsis:
        fail("`winner` still returns `...`, the placeholder.")
    return got


@test("legal_moves", "legal_moves lists the empty cells")
def _(L):
    legal = learner_fn(L, "legal_moves")
    for board in ([X, O, X, 0, O, 0, 0, X, 0], EMPTY, [X, O, X, O, X, O, 0, 0, 0]):
        got = legal(list(board))
        if got is Ellipsis:
            fail("`legal_moves` still returns `...`, the placeholder.")
        if got is None:
            fail("`legal_moves` returns nothing. It has to return a list of cell indices.")
        try:
            got = list(got)
        except TypeError:
            fail(f"`legal_moves` returns {type(got).__name__}, not a list of cell indices.")
        want = ref_moves(board)
        if got == want:
            continue
        taken = [i for i in range(9) if board[i] != 0]
        if got == taken:
            fail(f"On `{show(board)}` your list is {got}: those cells are already taken. Which value marks an empty cell?")
        if got == [0] * len(want):
            fail(f"On `{show(board)}` your list is {got}, the *contents* of the empty cells, which are all 0. "
                 f"A move is a cell's position, 0 to 8, so this board's moves are {want}.")
        if 8 in want and got == [i for i in want if i != 8]:
            fail(f"On `{show(board)}` cell 8, bottom right, is empty but isn't in your list. Which cells does `range(8)` visit?")
        fail(f"On `{show(board)}` your `legal_moves` gives {got}, but the empty cells are {want}.")


@test("ongoing_is_none", "winner is None while the game is still on")
def _(L):
    e = call_winner(L, EMPTY)
    m = call_winner(L, SPREAD)
    if e is None and m is None:
        return
    if e is False or m is False:
        fail("`winner` returns False for a game that's still on. Return None: 0 already means a draw, "
             "and in Python `False == 0`, so the two would get mixed up.")
    story = mixed_full_line(SPREAD, m)
    if story:
        fail(story)
    f = call_winner(L, DRAW)
    if e == 0 and m is None:
        fail("On an empty board `winner` says 0, a draw, before anyone has moved. Your loop finds the top row first: "
             "three equal cells, all of them empty. What else does a line need before it counts as a win?")
    if e == 0 and m == 0 and f is None:
        fail("Your draw check is upside down: an empty board comes back 0 (a draw) and a full board with no line "
             "comes back None. When is a game a draw?")
    if e == 0 and m == 0:
        fail(f"`winner` says 0, a draw, for an empty board and for `{show(SPREAD)}`, where 6 cells are still free. "
             "A draw needs a full board. How can you tell whether any cell is still 0?")
    fail(f"For the empty board `winner` returns {e!r}, and for `{show(SPREAD)}` {m!r}. A game that's still on should give None.")


@test("wins_for_both", "Three in a row wins, for X and for O")
def _(L):
    for board, who, line in WINS:
        got = call_winner(L, board)
        if same(got, who):
            continue
        if got is True:
            fail(f"On `{show(board)}` `winner` returns True. The game needs to know *who* won, "
                 "so return the player whose line it is: 1 or −1.")
        story = mixed_full_line(board, got)
        if story:
            fail(story)
        if who == O and got == X:
            fail(f"On `{show(board)}` O (−1) owns {line}, but `winner` says 1. Which cell's value tells you who owns the line?")
        if who == O and got is None:
            fail(f"On `{show(board)}` O (−1) owns {line}, but `winner` says None. X's lines work: does your check work for −1 too?")
        fail(f"On `{show(board)}` {name(who)} owns {line}, but `winner` returns {got!r}.")


@test("full_board_draw", "A full board with no line is a draw, 0")
def _(L):
    got = call_winner(L, DRAW)
    if not same(got, 0):
        if got is None:
            fail(f"On the full board `{show(DRAW)}` nobody has a line and no cell is free, but `winner` says None, "
                 "so this game would never end. What should it return when the board is full?")
        fail(f"On the full board `{show(DRAW)}` nobody has a line, so it's a draw, 0. `winner` returns {got!r}.")
    got = call_winner(L, FULL_WIN)
    if not same(got, X):
        if same(got, 0):
            fail(f"On `{show(FULL_WIN)}` X filled the last free cell and completed the 0-4-8 diagonal, but `winner` says 0. "
                 "A full board can still have a winner: check the lines first.")
        fail(f"On `{show(FULL_WIN)}` X owns the 0-4-8 diagonal, but `winner` returns {got!r}.")


@test("every_position", "Agrees with the rules on all 5,478 reachable positions")
def _(L):
    legal = learner_fn(L, "legal_moves")
    win = learner_fn(L, "winner")
    for board in reachable():
        got = win(list(board))
        if got is Ellipsis:
            fail("`winner` still returns `...`, the placeholder.")
        want = ref_winner(board)
        if not same(got, want):
            what = "the game is still on (None)" if want is None else "it's a draw (0)" if want == 0 else f"{name(want)} has won"
            fail(f"On `{show(board)}` `winner` returns {got!r}, but {what}.")
        if want is None:
            moves = legal(list(board))
            if moves is Ellipsis or moves is None:
                fail("`legal_moves` returns nothing yet. Is it still the `...` placeholder?")
            moves = list(moves)
            if moves != ref_moves(board):
                fail(f"On `{show(board)}` `legal_moves` gives {moves}, but the empty cells are {ref_moves(board)}.")
