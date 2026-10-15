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
    r, c = row + dr, col + dc
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


class Node:
    def __init__(self, prior):
        self.prior = prior
        self.n = 0
        self.w = 0.0
        self.children = {}

    def q(self):
        return self.w / self.n if self.n else 0.0


def select_child(node, c_puct):
    best_score, best_move, best_child = -math.inf, None, None
    for move, child in node.children.items():
        score = child.q() + c_puct * child.prior * math.sqrt(node.n) / (1 + child.n)
        if score > best_score:
            best_score, best_move, best_child = score, move, child
    return best_move, best_child


def backup(path, value):
    for node in reversed(path):
        value = -value
        node.n += 1
        node.w += value


def run_mcts(board, player, n_sims, rng, c_puct=1.5, rollout_fn=None):
    """S1's search, on one shared board: every move it plays, it takes back. board comes back unchanged."""
    rollout_fn = rollout_fn or rollout
    root = Node(1.0)
    for _ in range(n_sims):
        node, to_move, path, moves_made = root, player, [root], []
        value = None
        while node.children:
            move, node = select_child(node, c_puct)
            r = drop(board, move, to_move)
            moves_made.append(move)
            path.append(node)
            if is_win(board, r, move):
                value = -1      # the player to move here has just lost
                break
            to_move = -to_move
        if value is None:
            moves = legal_moves(board)
            if moves:
                for m in moves:
                    node.children[m] = Node(1 / len(moves))
                value = rollout_fn(board, to_move, rng)
            else:
                value = 0       # a full board is a draw
        for move in reversed(moves_made):
            undo(board, move)
        backup(path, value)
    return root


import copy


def any_win(board):
    """Check every cell for four in a row. Returns the winner, or 0."""
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] != 0 and is_win(board, r, c):
                return board[r][c]
    return 0


def rollout_copy(board, player, rng):
    """The slow way: copy the board, then check all of it after every move."""
    board = copy.deepcopy(board)
    turn = player
    while any(cell == 0 for cell in board[0]):
        drop(board, rng.choice(legal_moves(board)), turn)
        won = any_win(board)
        if won:
            return won * player
        turn = -turn
    return 0


def undo(board, col):
    # TODO: take the top piece back out of column col, the reverse of drop
    for row in range(ROWS):
        if board[row][col] != 0:
            board[row][col] = 0
            return


def rollout(board, player, rng):
    # TODO: play random legal moves on board itself, player first, until someone wins or the
    # board is full. Then undo every move you made, newest first, so board is exactly as you
    # found it. Return the result for player: 1 a win, -1 a loss, 0 a draw.
    played = []
    p = player
    result = 0
    while result == 0 and legal_moves(board):
        col = rng.choice(legal_moves(board))
        row = drop(board, col, p)
        played.append(col)
        if is_win(board, row, col):
            result = 1 if p == player else -1
        p = -p
    for col in reversed(played):
        undo(board, col)
    return result


rng = random.Random(0)
for fn in (rollout_copy, rollout):
    start = time.perf_counter()
    root = run_mcts(new_board(), 1, 300, rng, rollout_fn=fn)
    rate = 300 / (time.perf_counter() - start)
    print(f"{fn.__name__:<13} {rate:6.0f} sims/s")


def visit_policy(root, temperature=1.0):
    """S1's visit_policy, with 7 columns instead of 9 cells."""
    visits = [root.children[m].n if m in root.children else 0 for m in range(COLS)]
    if temperature == 0:
        best = visits.index(max(visits))
        return [1.0 if m == best else 0.0 for m in range(COLS)]
    powered = [n ** (1 / temperature) for n in visits]
    total = sum(powered)
    return [x / total for x in powered]


def snapshot(board, player):
    # TODO: a copy of board as player sees it: player's own pieces 1, the opponent's -1.
    # A real copy, so the moves played after it can't change it.
    return [[cell * player for cell in row] for row in board]


def to_examples(history, result):
    # history holds one (state, player, pi) per move. result: 1 if X won, -1 if O won, 0 a draw.
    # TODO: one (state, pi, z) per move, where z is the final result from the view of
    # the player who was to move in that state
    return [(state, pi, result * player) for state, player, pi in history]


def self_play_game(n_sims, rng, explore_moves=8):
    """One game of the search against itself. Returns its training examples."""
    board = new_board()
    player = 1
    history = []
    while True:
        root = run_mcts(board, player, n_sims, rng)
        # sample from the visits early on for varied openings, then play the most visited move
        pi = visit_policy(root, 1.0 if len(history) < explore_moves else 0)
        history.append((snapshot(board, player), player, pi))
        col = rng.choices(range(COLS), weights=pi)[0]
        row = drop(board, col, player)
        if is_win(board, row, col):
            return to_examples(history, player)
        if not legal_moves(board):
            return to_examples(history, 0)
        player = -player


examples = self_play_game(30, random.Random(1))
state, pi, z = examples[-1]
print(f"{len(examples)} examples. The last mover's z: {z}, the move before: {examples[-2][2]}")
print("The last state, from the last mover's side, so their pieces show as X:")
show(state)


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


def negamax(board, depth, player):
    """(score, col): the best column for player, looking depth moves ahead. Scores are from
    player's view. A win scores 1000 + depth, so a sooner win scores higher."""
    moves = legal_moves(board)
    if depth == 0 or not moves:
        return heuristic(board, player), None
    best_score, best_col = -math.inf, None
    for col in moves:
        # TODO: drop player's piece into col. It scores 1000 + depth if it wins on the spot;
        # otherwise minus the opponent's best score one level deeper. Then take it back out.
        score = ...
        if score > best_score:
            best_score, best_col = score, col
    return best_score, best_col


def minimax_bot(board, player):
    return negamax(board, 4, player)[1]


puzzle = new_board()
for col, player in [(2, 1), (1, -1), (3, 1), (3, -1), (4, 1)]:
    drop(puzzle, col, player)
show(puzzle)
print("If X were to move, depth 1:", negamax(puzzle, 1, 1))
print("O to move, the depth-4 bot plays column", minimax_bot(puzzle, -1))
