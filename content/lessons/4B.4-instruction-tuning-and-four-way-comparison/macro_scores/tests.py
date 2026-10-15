from collections import namedtuple
from harness import test, fail, learner_fn

Row = namedtuple("Row", "model task item correct tokens")
# Two models on four tasks of different sizes (6, 3, 2 and 2 items). Four tasks, so code that
# divides by 3 is caught, and unequal sizes, so pooling gives a different number.
SPEC = {
    "a": {"t1": [1, 1, 1, 1, 1, 0], "t2": [1, 1, 0], "t3": [0, 0], "t4": [1, 0]},
    "b": {"t1": [1, 0, 0, 1, 0, 0], "t2": [1, 0, 0], "t3": [1, 1], "t4": [1, 1]},
}
ROWS_T = [Row(m, t, i + 1, c, 10 + i) for m, tasks in SPEC.items() for t, cs in tasks.items() for i, c in enumerate(cs)]


def macro(spec):
    return {m: 100 * sum(sum(cs) / len(cs) for cs in tasks.values()) / len(tasks) for m, tasks in spec.items()}


def pooled(spec):
    return {m: 100 * sum(sum(cs) for cs in tasks.values()) / sum(len(cs) for cs in tasks.values()) for m, tasks in spec.items()}


def call(L):
    fn = learner_fn(L, "macro_scores")
    out = fn(list(ROWS_T))
    if out is None:
        fail("`macro_scores` returns nothing. It has to return the `scores` dict.")
    if out is Ellipsis or (isinstance(out, dict) and any(v is Ellipsis for v in out.values())):
        fail("Your scores are still `...`, the placeholders. Fill in `acc`, `mine` and the score line.")
    if not isinstance(out, dict):
        fail(f"`macro_scores` should return a dict of model → score, but it returns {type(out).__name__}.")
    if set(out) != set(SPEC):
        fail(f"On rows for models {sorted(SPEC)}, `macro_scores` returns scores for {sorted(out)}. One score per model.")
    for m, v in out.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            fail(f"The score for `{m}` should be a number, but it's {type(v).__name__}.")
    return {m: float(v) for m, v in out.items()}


def near(a, b):
    return all(abs(a[m] - b[m]) < 1e-6 for m in b)


@test("one_score_per_model", "macro_scores gives every model one score in %")
def _(L):
    got = call(L)
    want = macro(SPEC)
    if near(got, {m: v / 100 for m, v in want.items()}) or (max(got.values()) <= 1 and not near(got, want)):
        fail(f"Your scores are between 0 and 1 ({', '.join(f'{m}: {v:.3f}' for m, v in got.items())}). "
             "That's a fraction. The table shows percentages, so multiply by 100.")


@test("tasks_count_equally", "A model's score is the mean of its task accuracies")
def _(L):
    got = call(L)
    want = macro(SPEC)
    if near(got, want):
        return
    shown = ", ".join(f"{m}: {v:.1f}%" for m, v in got.items())
    if near(got, pooled(SPEC)):
        fail(f"You pooled every item, so you get {shown}. Pooled, t1's 6 items count 3 times as much as t3's 2, and model b, "
             f"which aces the two small tasks, comes out behind. Score each task first, then average the tasks.")
    if len(set(round(v, 6) for v in got.values())) == 1 and abs(want["a"] - want["b"]) > 1:
        fail(f"Every model gets the same score, {next(iter(got.values())):.1f}%: `mine` takes the accuracies of every model, "
             "not just this one. Keep the keys whose model is `model`.")
    count = {m: 100 * sum(sum(cs) for cs in tasks.values()) / len(tasks) for m, tasks in SPEC.items()}
    if near(got, count):
        fail(f"Your scores ({shown}) are far over 100%: `acc` counts right answers instead of the share of right answers. "
             "Divide each group's right answers by the group's size.")
    n = len(ROWS_T)
    by_rows = {m: 100 * sum(sum(cs) / n for cs in tasks.values()) / len(tasks) for m, tasks in SPEC.items()}
    if near(got, by_rows):
        fail(f"Your scores ({shown}) are tiny: each task's right answers are divided by every row in the eval, all models and tasks. "
             "Divide by the size of that one (model, task) group.")
    for k in (2, 3):
        fixed = {m: 100 * sum(sum(cs) / len(cs) for cs in tasks.values()) / k for m, tasks in SPEC.items()}
        if near(got, fixed):
            fail(f"On an eval with 4 tasks you divide by {k} tasks, so the scores come out too high ({shown}). "
                 "That fits 4A.5's eval, but not one with more tasks. Count this model's own tasks with `len(mine)`.")
    fail(f"With these rows the macro scores are {', '.join(f'{m}: {v:.1f}%' for m, v in want.items())}, but you get {shown}.")
