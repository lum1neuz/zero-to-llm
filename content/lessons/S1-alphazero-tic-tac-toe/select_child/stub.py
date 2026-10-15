

def select_child(node, c_puct=1.5):
    best_score, best_move, best_child = -math.inf, None, None
    for move, child in node.children.items():
        # TODO: the PUCT score Q + c·P·√N / (1 + n). Q and n are the child's,
        # P is child.prior, N is the parent's visit count
        score = ...
        if score > best_score:
            best_score, best_move, best_child = score, move, child
    return best_move, best_child


root = Node(1.0)
root.n = 10
root.children = {0: Node(0.5), 4: Node(0.5)}
root.children[0].n, root.children[0].w = 9, 0.9
print("select_child picks move", select_child(root)[0])
