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
        score = child.q() + c_puct * child.prior * math.sqrt(node.n / (1 + child.n))
        if score > best_score:
            best_score, best_move, best_child = score, move, child
    return best_move, best_child


root = Node(1.0)
root.n = 10
root.children = {0: Node(0.5), 4: Node(0.5)}
root.children[0].n, root.children[0].w = 9, 0.9
print("select_child picks move", select_child(root)[0])
