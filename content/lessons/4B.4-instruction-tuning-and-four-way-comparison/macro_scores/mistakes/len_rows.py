from collections import namedtuple

# A slice of your 4A.5 eval: the same 12 items for all four models. One row per answer: the model,
# the task, the item, whether it was right (1 or 0), and the answer's length in that model's tokens.
#   mn_qa  short answer, exact match   "Монгол Улсын нийслэл хаана байдаг вэ?" → "Улаанбаатар"
#   mn_mc  four options                "Би сургууль ___ явна."  A) руу  B) рүү  C) луу  D) лүү → A
Row = namedtuple("Row", "model task item correct tokens")
ROWS = [Row(*r) for r in [
    ("base", "mn_qa", 1, 0, 12), ("base", "mn_qa", 2, 0, 9), ("base", "mn_qa", 3, 0, 14), ("base", "mn_qa", 4, 0, 11),
    ("base", "mn_mc", 1, 1, 8), ("base", "mn_mc", 2, 0, 11), ("base", "mn_mc", 3, 1, 7), ("base", "mn_mc", 4, 0, 10),
    ("base", "mn_mc", 5, 1, 9), ("base", "mn_mc", 6, 0, 12), ("base", "mn_mc", 7, 1, 6), ("base", "mn_mc", 8, 0, 9),
    ("control", "mn_qa", 1, 1, 10), ("control", "mn_qa", 2, 0, 9), ("control", "mn_qa", 3, 1, 13), ("control", "mn_qa", 4, 0, 8),
    ("control", "mn_mc", 1, 1, 8), ("control", "mn_mc", 2, 1, 10), ("control", "mn_mc", 3, 1, 7), ("control", "mn_mc", 4, 0, 11),
    ("control", "mn_mc", 5, 1, 9), ("control", "mn_mc", 6, 1, 12), ("control", "mn_mc", 7, 0, 6), ("control", "mn_mc", 8, 1, 9),
    ("extended", "mn_qa", 1, 0, 6), ("extended", "mn_qa", 2, 1, 5), ("extended", "mn_qa", 3, 0, 7), ("extended", "mn_qa", 4, 0, 5),
    ("extended", "mn_mc", 1, 1, 4), ("extended", "mn_mc", 2, 0, 6), ("extended", "mn_mc", 3, 1, 4), ("extended", "mn_mc", 4, 1, 5),
    ("extended", "mn_mc", 5, 0, 5), ("extended", "mn_mc", 6, 1, 6), ("extended", "mn_mc", 7, 0, 3), ("extended", "mn_mc", 8, 1, 5),
    ("extended_it", "mn_qa", 1, 1, 5), ("extended_it", "mn_qa", 2, 1, 5), ("extended_it", "mn_qa", 3, 1, 6), ("extended_it", "mn_qa", 4, 0, 4),
    ("extended_it", "mn_mc", 1, 1, 4), ("extended_it", "mn_mc", 2, 1, 5), ("extended_it", "mn_mc", 3, 0, 4), ("extended_it", "mn_mc", 4, 1, 6),
    ("extended_it", "mn_mc", 5, 1, 5), ("extended_it", "mn_mc", 6, 0, 6), ("extended_it", "mn_mc", 7, 0, 3), ("extended_it", "mn_mc", 8, 1, 5),
]]


def by(rows, *fields):
    """Group rows by one field or several: by(ROWS, "model") → {"base": [rows], …},
    by(ROWS, "model", "task") → {("base", "mn_qa"): [rows], …}."""
    groups = {}
    for r in rows:
        key = getattr(r, fields[0]) if len(fields) == 1 else tuple(getattr(r, f) for f in fields)
        groups.setdefault(key, []).append(r)
    return groups
def macro_scores(rows):
    # TODO: the accuracy of each (model, task) pair, from 0 to 1
    acc = {key: sum(r.correct for r in group) / len(rows) for key, group in by(rows, "model", "task").items()}
    scores = {}
    for model in sorted({r.model for r in rows}):
        # TODO: this model's task accuracies
        mine = [a for (m, t), a in acc.items() if m == model]
        # TODO: their mean, in %
        scores[model] = 100 * sum(mine) / len(mine)
    return scores


for model, score in macro_scores(ROWS).items():
    print(f"{model:<12} {score:5.1f}%")
