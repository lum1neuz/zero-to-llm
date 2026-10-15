import math
import random
from harness import test, fail, learner_fn, LearnerError


class FakeNode:
    """Same fields as the lesson's Node, so the tests don't depend on yours."""

    def __init__(self, prior, n=0, w=0.0):
        self.prior, self.n, self.w, self.children = prior, n, w, {}

    def q(self):
        return self.w / self.n if self.n else 0.0


def tree(parent_n, kids):
    """A root with N = parent_n and children {move: (P, n, w)}."""
    root = FakeNode(1.0, parent_n)
    root.children = {m: FakeNode(*k) for m, k in kids.items()}
    return root


def puct(node, child, c):
    return child.q() + c * child.prior * math.sqrt(node.n) / (1 + child.n)


def ref_pick(root, c):
    return max(root.children, key=lambda m: puct(root, root.children[m], c))


def pick(L, root, c=1.5):
    select = learner_fn(L, "select_child")
    try:
        out = select(root, c)
    except LearnerError as e:
        if "ZeroDivisionError" in str(e):
            fail("`select_child` crashed with a division by zero. Some children have never been visited, so their n is 0. "
                 "Anything you divide by n has to cope with that: `1 + n` does it for the exploration term, "
                 "and `child.q()` already does it for Q.")
        raise
    if not (isinstance(out, tuple) and len(out) == 2):
        fail("`select_child` should return `(move, child)`. Keep the `return best_move, best_child` line at the end.")
    move, child = out
    if move not in root.children or child is not root.children[move]:
        fail(f"`select_child` returned move {move!r}, which isn't one of this node's children {sorted(root.children)}.")
    return move


def kids_txt(root):
    return ", ".join(f"move {m} (P {k.prior:.2f}, n {k.n}, Q {k.q():.2f})".replace("-", "−") for m, k in root.children.items())


@test("explores_unvisited", "An unvisited move with a decent prior gets tried")
def _(L):
    root = tree(10, {0: (0.5, 9, 0.9), 4: (0.5, 0, 0.0)})
    got = pick(L, root)
    if got == 4:
        return
    fail("Move 4 has never been visited, but `select_child` picks move 0, which has been tried 9 of 10 times for Q = 0.1. "
         "With your score, how big is the exploration bonus of a child with n = 0? "
         "The N under the square root is the parent's visit count, `node.n`.")


@test("prior_orders_unvisited", "Among unvisited moves, the bigger prior goes first")
def _(L):
    root = tree(3, {1: (0.1, 0, 0.0), 5: (0.6, 0, 0.0), 7: (0.3, 0, 0.0)})
    got = pick(L, root)
    if got == 5:
        return
    p = root.children[got].prior
    fail(f"All three moves are unvisited, so only their priors P tell them apart. Move 5 has P = 0.6, "
         f"but you picked move {got}, with P = {p:.1f}. Where does P go in the score?")


@test("exploits_good_moves", "Once explored, the better Q wins")
def _(L):
    root = tree(100, {2: (0.5, 50, 40.0), 6: (0.5, 49, -9.8)})
    got = pick(L, root)
    if got == 2:
        return
    greedy = pick(L, root, 0.0)
    if greedy == 6:
        fail("Move 2 has Q = 0.8 after 50 visits and move 6 has Q = −0.2 after 49, with the same prior. "
             "You picked move 6, the one with the *lowest* Q. Q is from the view of the player choosing at this node, "
             "so a higher Q is better for them.")
    fail("Move 2 has Q = 0.8 after 50 visits and move 6 has Q = −0.2 after 49, with the same prior. "
         "You picked move 6. Which part of the score knows how well a move has gone so far?")


@test("average_not_total", "Q is the average value per visit")
def _(L):
    root = tree(91, {0: (0.5, 80, 40.0), 8: (0.5, 10, 8.0)})
    got = pick(L, root)
    if got == 8:
        return
    fail("Move 0 has w = 40 over 80 visits (Q = 0.5), move 8 has w = 8 over 10 visits (Q = 0.8). "
         "You picked move 0: its *total* is bigger only because it was visited 8 times as often. "
         "Which number says how good a move is per visit?")


@test("uses_c_puct", "c_puct sets how much to explore")
def _(L):
    root = tree(10, {0: (0.5, 9, 0.9), 4: (0.5, 0, 0.0)})
    got = pick(L, root, 0.0)
    if got == 0:
        return
    fail("With `c_puct=0` there's no exploration bonus at all, so the pick should be the best Q: move 0 (Q = 0.1) "
         "over move 4 (Q = 0, unvisited). You still picked move 4. Is your score using the `c_puct` it's given?")


@test("matches_puct", "Matches PUCT on 200 random nodes")
def _(L):
    rng = random.Random(7)
    for _ in range(200):
        k = rng.randint(2, 6)
        moves = rng.sample(range(9), k)
        priors = [rng.random() + 0.05 for _ in moves]
        kids = {}
        for m, p in zip(moves, priors):
            n = rng.randint(0, 30)
            kids[m] = (p / sum(priors), n, rng.uniform(-1, 1) * n)
        root = tree(sum(v[1] for v in kids.values()) + 1, kids)
        c = rng.choice([0.5, 1.0, 1.5, 3.0])
        want = ref_pick(root, c)
        got = pick(L, root, c)
        if got != want:
            fail(f"Parent N = {root.n}, c_puct = {c}. Children: {kids_txt(root)}. "
                 f"PUCT picks move {want}, yours picks move {got}. Work out both scores by hand: Q + c·P·√N / (1 + n).")
