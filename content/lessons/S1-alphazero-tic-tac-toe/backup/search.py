

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
