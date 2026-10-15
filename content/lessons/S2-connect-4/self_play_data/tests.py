from harness import test, fail, learner_fn, script_error

ROWS, COLS = 6, 7


def board_of(*rows):
    rows = ["......."] * (ROWS - len(rows)) + list(rows)
    return [[{".": 0, "X": 1, "O": -1}[ch] for ch in row] for row in rows]


def txt(board):
    return "`" + " / ".join("".join(".XO"[v] for v in row) for row in board) + "`"


POSITION = board_of("...O...", "..XXO..")  # X at (5, 2) and (5, 3); O at (5, 4) and (4, 3)


def snap(L, board, player):
    fn = learner_fn(L, "snapshot")
    got = fn(board, player)
    if got is Ellipsis:
        fail("`snapshot` still returns `...`, the placeholder.")
    if got is None:
        fail("`snapshot` returns nothing. Return the copied board.")
    return got


def shape_ok(state):
    return isinstance(state, list) and len(state) == ROWS and all(isinstance(r, list) and len(r) == COLS for r in state)


@test("snapshot_is_copy", "snapshot is a copy that later moves don't change")
def _(L):
    board = [row[:] for row in POSITION]
    state = snap(L, board, 1)
    if not shape_ok(state):
        fail("`snapshot` should return 6 rows of 7 cells, the same shape as the board.")
    before = [row[:] for row in state]
    board[3][3] = 1  # the next move lands on the live board
    if state is board:
        fail("`snapshot` returns the live board itself. self_play_game keeps playing on that board, so by the end of the game "
             "every stored state would show the final position. Return a new list.")
    if state != before:
        fail("`snapshot` makes a new outer list, but the rows inside it are still the board's own rows, "
             "so the next move shows up in your stored state too. Build new rows as well.")


@test("snapshot_view", "snapshot shows the player to move as 1")
def _(L):
    as_o = snap(L, [row[:] for row in POSITION], -1)
    as_x = snap(L, [row[:] for row in POSITION], 1)
    flipped = [[-v for v in row] for row in POSITION]
    if as_x == POSITION and as_o == flipped:
        return
    if as_o == POSITION:
        fail("With O to move, your snapshot still shows O's pieces as −1. The net sees every position from the mover's side, "
             "own pieces 1 and the opponent's −1, so one set of weights can play both colours.")
    if as_x == flipped:
        fail("With X to move, your snapshot flips X's pieces to −1. Multiply by the player instead: X is 1, so X's view is "
             "the board as it is, and O is −1, which flips it.")
    fail(f"For {txt(POSITION)}, X's view should be the board as it is, and O's view the board with every piece's sign flipped.")


def history_of(players):
    """A fake game: distinct state and pi objects for each move."""
    return [([[i] * COLS for _ in range(ROWS)], p, [i / 10] * COLS) for i, p in enumerate(players)]


def examples(L, history, result):
    fn = learner_fn(L, "to_examples")
    got = fn(history, result)
    if got is Ellipsis:
        fail("`to_examples` still returns `...`, the placeholder.")
    if got is None:
        fail("`to_examples` returns nothing. Return a list with one `(state, pi, z)` per move.")
    got = list(got)
    if len(got) != len(history):
        fail(f"The game had {len(history)} moves, and `to_examples` returns {len(got)} examples. It needs one per move.")
    return got


@test("example_shape", "to_examples gives one (state, pi, z) per move")
def _(L):
    history = history_of([1, -1, 1, -1, 1])
    got = examples(L, history, 1)
    for i, (ex, (state, player, pi)) in enumerate(zip(got, history)):
        if not (isinstance(ex, tuple) and len(ex) == 3):
            fail(f"Example {i} is {ex!r}. Each example should be a tuple of three: `(state, pi, z)`.")
        if ex[0] is not state:
            fail(f"Example {i}'s first item isn't move {i}'s state. Each example starts with its own move's state.")
        if ex[1] is not pi:
            what = "the player" if ex[1] == player else "something else"
            fail(f"Example {i}'s second item is {what}, not move {i}'s pi. The order is `(state, pi, z)`: the net learns "
                 "pi from the policy head and z from the value head, and the player is already folded into the state.")


@test("z_from_mover", "z is the final result from each mover's view")
def _(L):
    for result, players in ((1, [1, -1, 1, -1, 1]), (-1, [1, -1, 1, -1])):
        got = [ex[2] for ex in examples(L, history_of(players), result)]
        want = [result * p for p in players]
        if got == want:
            continue
        winner = "X" if result == 1 else "O"
        loser = "O" if result == 1 else "X"
        if got == [result] * len(players):
            fail(f"{winner} won, and every example got z = {result:+d}, {loser}'s positions too. z is the result from the view "
                 f"of the player who was to move: for {loser}, {winner}'s win is a loss, −1.".replace("-", "−"))
        if got == [-w for w in want]:
            fail(f"Every z is flipped: {winner} won, but {winner}'s positions got z = −1. z is the result from the view of the "
                 "player who was to move, so the winner's positions get +1.")
        fail(f"{winner} won a {len(players)}-move game whose movers were {players}. z should be {want}; yours is {got}.")


@test("draw_is_zero", "A drawn game labels every position 0")
def _(L):
    got = [ex[2] for ex in examples(L, history_of([1, -1, 1, -1]), 0)]
    if got == [0, 0, 0, 0]:
        return
    if got == [-1, -1, -1, -1]:
        fail("In a drawn game your examples get z = −1: a draw counted as a loss for both sides. "
             "The value head should learn that a drawn position is worth 0. What's the result times the player when the result is 0?")
    fail(f"In a drawn game every position should get z = 0. Yours are {got}.")


@test("self_play_runs", "One self-play game runs and labels every move")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "examples" not in L:
        fail("The self-play game at the bottom is missing. Put the `examples = self_play_game(...)` line back.")
    zs = [ex[2] for ex in L["examples"]]
    if zs and zs[-1] not in (0, 1):
        fail(f"The last move of a game either won it or filled the board, so its z is 1 or 0. Yours is {zs[-1]}.")
