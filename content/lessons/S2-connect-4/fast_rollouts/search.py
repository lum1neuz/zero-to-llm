

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
