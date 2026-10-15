

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
