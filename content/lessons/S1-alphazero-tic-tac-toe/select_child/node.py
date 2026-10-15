

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
