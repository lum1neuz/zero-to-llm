import math
import random

# A board is a list of 9 cells, left to right, top to bottom:
#    0 | 1 | 2
#    3 | 4 | 5
#    6 | 7 | 8
# 1 is X, -1 is O, 0 is empty. X moves first.

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def legal_moves(board):
    # TODO: the indices of the empty cells, smallest first
    return [i for i in range(9) if board[i] == 0]


def winner(board):
    # TODO: 1 or -1 if that player owns all three cells of a line in LINES,
    # 0 if the board is full and nobody does (a draw), None while the game goes on
    for a, b, c in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0 if 0 not in board else None


board = [1, -1, 1,
         0, -1, 0,
         0, 1, 0]
print("legal moves:", legal_moves(board))
print("winner:", winner(board))


class Node:
    """One position in the search tree, reached by one move."""

    def __init__(self, prior):
        self.prior = prior    # P: how much the prior likes the move into this node
        self.n = 0            # visits
        self.w = 0.0          # total value, for the player who made the move into this node
        self.children = {}    # move -> Node

    def q(self):
        # Q: the average value, and 0 for a node nobody has visited yet
        return self.w / self.n if self.n else 0.0


def select_child(node, c_puct=1.5):
    best_score, best_move, best_child = -math.inf, None, None
    for move, child in node.children.items():
        # TODO: the PUCT score Q + c·P·√N / (1 + n). Q and n are the child's,
        # P is child.prior, N is the parent's visit count
        score = child.q() + c_puct * child.prior * math.sqrt(node.n) / (1 + child.n)
        if score > best_score:
            best_score, best_move, best_child = score, move, child
    return best_move, best_child


root = Node(1.0)
root.n = 10
root.children = {0: Node(0.5), 4: Node(0.5)}
root.children[0].n, root.children[0].w = 9, 0.9
print("select_child picks move", select_child(root)[0])


def rollout(board, player, rng):
    """Random moves to the end of the game. Returns the result for `player`."""
    board = board[:]
    turn = player
    while winner(board) is None:
        board[rng.choice(legal_moves(board))] = turn
        turn = -turn
    return winner(board) * player


def run_mcts(board, player, n_sims, rng, c_puct=1.5):
    """Search from `board` with `player` to move. Returns the root of the tree."""
    root = Node(1.0)
    for _ in range(n_sims):
        node, b, to_move, path = root, board[:], player, [root]
        # 1. select: follow PUCT down to a leaf
        while node.children:
            move, node = select_child(node, c_puct)
            b[move] = to_move
            to_move = -to_move
            path.append(node)
        # 2. expand and evaluate: a uniform prior over the legal moves, a random rollout for the value
        result = winner(b)
        if result is None:
            moves = legal_moves(b)
            for m in moves:
                node.children[m] = Node(1 / len(moves))
            value = rollout(b, to_move, rng)
        else:
            value = result * to_move
        # 3. back up: value is the result for to_move, the player to move at the leaf
        backup(path, value)
    return root


def backup(path, value):
    # path runs from the root to the leaf. value is the result for the player to move
    # at the leaf: 1 a win, -1 a loss, 0 a draw. Each node's w counts for the player
    # who made the move INTO that node.
    # TODO: walk from the leaf back up to the root, adding a visit and the value to each node
    for node in reversed(path):
        value = -value
        node.n += 1
        node.w += value


board = [1, 1, 0,
         -1, -1, 0,
         0, 0, 0]
root = run_mcts(board, 1, 200, random.Random(0))
print("X to move. Visits per move:", {m: c.n for m, c in sorted(root.children.items())})


from functools import lru_cache


@lru_cache(maxsize=None)
def solve(cells, player):
    """(value, move) for `player` to move, when both sides play perfectly."""
    result = winner(list(cells))
    if result is not None:
        return result * player, None
    best = (-2, None)
    for m in legal_moves(list(cells)):
        after = cells[:m] + (player,) + cells[m + 1:]
        v = -solve(after, -player)[0]
        if v > best[0]:
            best = (v, m)
    return best


def perfect_move(board, player):
    return solve(tuple(board), player)[1]


def visit_policy(root, temperature=1.0):
    # TODO: 9 probabilities, one per cell, from the visit counts n of the root's children.
    # A cell with no child gets 0. Each move's share is proportional to n ** (1 / temperature):
    # temperature 1 follows the visits, lower is sharper, and temperature 0 puts all of it
    # on the most visited move.
    visits = [root.children[m].n if m in root.children else 0 for m in range(9)]
    powered = [n ** (1 / temperature) for n in visits]
    total = sum(powered)
    return [x / total for x in powered]


def play_perfect(mcts_player, sims, rng):
    """One game: your MCTS plays mcts_player, the perfect player plays the other side."""
    board = [0] * 9
    player = 1
    while winner(board) is None:
        if player == mcts_player:
            pi = visit_policy(run_mcts(board, player, sims, rng), temperature=0)
            move = pi.index(max(pi))
        else:
            move = perfect_move(board, player)
        board[move] = player
        player = -player
    return winner(board)


SIMS = 200
GAMES = 10
rng = random.Random(0)
pi = visit_policy(run_mcts([0] * 9, 1, SIMS, rng))
print("Policy target from the empty board:", [round(p, 2) for p in pi])
results = [play_perfect(1 if g % 2 == 0 else -1, SIMS, rng) for g in range(GAMES)]
draws = results.count(0)
print(f"{SIMS} sims: {draws}/{GAMES} draws vs perfect play ({100 * draws / GAMES:.0f}%)")
