

def backup(path, value):
    # path runs from the root to the leaf. value is the result for the player to move
    # at the leaf: 1 a win, -1 a loss, 0 a draw. Each node's w counts for the player
    # who made the move INTO that node.
    # TODO: walk from the leaf back up to the root, adding a visit and the value to each node
    ...


board = [1, 1, 0,
         -1, -1, 0,
         0, 0, 0]
root = run_mcts(board, 1, 200, random.Random(0))
print("X to move. Visits per move:", {m: c.n for m, c in sorted(root.children.items())})
