import random
from harness import test, fail, learner_fn


class FakeNode:
    def __init__(self, prior=1.0):
        self.prior, self.n, self.w, self.children = prior, 0, 0.0, {}

    def q(self):
        return self.w / self.n if self.n else 0.0


def signs(ws):
    return "[" + ", ".join(f"{w:+g}".replace("-", "−") for w in ws) + "]"


def backed_up(L, k, values):
    """A fresh path of k nodes, root first, after backup(path, v) for each v."""
    backup = learner_fn(L, "backup")
    path = [FakeNode() for _ in range(k)]
    for v in values:
        backup(path, v)
    if all(node.n == 0 and node.w == 0 for node in path):
        fail("Nothing changed: every node still has n = 0 and w = 0. Is `backup` still the `...` placeholder?")
    return path


def want_w(k, value):
    # the leaf (index k - 1) gets -value, its parent +value, and so on up to the root
    return [-value if (k - 1 - i) % 2 == 0 else value for i in range(k)]


@test("counts_visits", "Every node on the path gains one visit")
def _(L):
    path = backed_up(L, 3, [1.0])
    ns = [node.n for node in path]
    ws = [node.w for node in path]
    if ns == [1, 1, 1]:
        return
    if ns == [0, 0, 0] and ws == [0, 0, 0]:
        fail("Nothing changed: every node still has n = 0 and w = 0. Is `backup` still the `...` placeholder?")
    if ns == [0, 0, 0]:
        fail(f"w changed ({signs(ws)} from root to leaf), but n is still 0 everywhere. Q is w / n, so every node needs its visit counted too.")
    if ns == [0, 0, 1]:
        fail("Only the leaf got a visit. This simulation walked through every node from the root down to the leaf, "
             "so each of them gets n += 1.")
    fail(f"After one backup on a 3-node path, the visit counts are {ns} from root to leaf. Each node should have exactly 1.")


@test("signs_alternate", "The value flips sign at every level, starting at the leaf")
def _(L):
    ws3 = [node.w for node in backed_up(L, 3, [1.0])]
    ws4 = [node.w for node in backed_up(L, 4, [1.0])]
    w3, w4 = want_w(3, 1.0), want_w(4, 1.0)
    if ws3 == w3 and ws4 == w4:
        return
    if len(set(ws4)) == 1:
        fail(f"After backing up value = +1, every node has w = {ws4[0]:+g}. But the levels alternate between X's moves and O's: "
             "a result that's good for one player is bad for the other. Flip the sign at each step up.".replace("-", "−"))
    if ws3 == [-w for w in w3] and ws4 == [-w for w in w4]:
        fail(f"After backing up value = +1 the leaf has w = +1 (yours, root to leaf: {signs(ws4)}). value is the result for "
             "the player *to move* at the leaf, but the leaf's w counts for the player who moved *into* it, their opponent. "
             "Flip before you add.")
    if ws3 == w3 and ws4 == [-w for w in w4]:
        fail("On a 3-node path your signs are right, but on a 4-node path they're all reversed. You walk from the root down, "
             "so the first flip lands on the root and the leaf's sign depends on the path's length. "
             "The sign you know is the leaf's: start there.")
    fail(f"After backing up value = +1 on a 4-node path, w from root to leaf should be {signs(w4)}. Yours is {signs(ws4)}.")


@test("adds_up", "A second backup adds to the first")
def _(L):
    path = backed_up(L, 3, [1.0, 1.0])
    leaf = path[-1]
    if leaf.n == 2 and leaf.w == -2.0:
        return
    if leaf.n == 2 and leaf.w == -1.0:
        fail("After two backups of the same result the leaf has n = 2 but w = −1, the second value only. "
             "w is a running total: add to it.")
    if leaf.n == 1:
        fail("After two backups the leaf still has n = 1. n is a running count: add 1 each time.")
    fail(f"After two backups of value = +1, the leaf should have n = 2 and w = −2. Yours has n = {leaf.n}, w = {leaf.w:g}.")


def most_visited(root):
    return max(root.children, key=lambda m: root.children[m].n)


@test("search_works", "Your MCTS takes the win and blocks the loss")
def _(L):
    search = learner_fn(L, "run_mcts")
    cases = [
        ([1, 1, 0, -1, -1, 0, 0, 0, 0], 1, 2, "X to move on `X X . / O O . / . . .` should win at 2"),
        ([1, 1, 0, 0, -1, 0, 0, 0, 0], -1, 2, "O to move on `X X . / . O . / . . .` has to block at 2"),
    ]
    for board, player, want, what in cases:
        root = search(list(board), player, 200, random.Random(0))
        total = sum(c.n for c in root.children.values())
        if total < 199:
            fail(f"After 200 simulations the root's children have {total} visits between them. Every simulation after the "
                 "first passes through one of them, so they should add up to 199. Check that the tests above pass.")
        got = most_visited(root)
        if got != want:
            visits = {m: c.n for m, c in sorted(root.children.items())}
            fail(f"{what}, but with 200 simulations your search visits move {got} most. Visits: {visits}. "
                 "Check that `select_child` from step 2 still passes its tests.")
