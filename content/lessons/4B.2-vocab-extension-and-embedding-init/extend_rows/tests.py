import numpy as np
import torch
from harness import test, fail, learner_fn, arr

# A 6-token vocab with 16-wide rows (as wide as the script's E and H, so code that reads the
# global E instead of its argument runs and gets caught on its values), and three new tokens with
# 2, 3 and 1 pieces.
W = torch.randn(6, 16, generator=torch.Generator().manual_seed(7))
PIECES = [[1, 2], [0, 3, 5], [4]]
V, N = 6, len(PIECES)


def expected(w, pieces):
    a = arr(w).astype(np.float64)
    return np.concatenate([a, np.stack([a[ids].mean(axis=0) for ids in pieces])])


def call(L):
    """Call the learner's extend, explaining the shape errors that cat and stack raise."""
    learner_fn(L, "extend")  # explains a missing or non-callable extend
    try:
        out = L["extend"](W.clone(), [list(p) for p in PIECES])
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "stack expects each tensor to be equal size" in msg or "must match except in dimension 0" in msg:
            fail("`mean(dim=1)` averages the 16 numbers inside each piece's row, so each new token becomes one number per piece "
                 "instead of one 16-wide row. You want the mean across the pieces. Which dimension of `weight[ids]` runs across them?")
        if "must match except in dimension 1" in msg:
            fail("`torch.cat(..., dim=1)` joins along `dim=1`, so it tries to put the new rows beside the old ones, as extra columns. "
                 "New tokens are new rows. Which dimension counts rows?")
        out = learner_fn(L, "extend")(W.clone(), [list(p) for p in PIECES])  # anything else: the harness explains it
    return check_out(out)


def check_out(out):
    if out is None:
        fail("`extend` returns nothing. It has to return the grown matrix.")
    if out is Ellipsis:
        fail("`extend` still returns `...`, the placeholder. Fill in `new_rows` and the return line.")
    a = arr(out)
    if a is None:
        fail(f"`extend` should return a tensor, but it returns {type(out).__name__}.")
    return a.astype(np.float64)


def grown(L):
    """call() for the tests after grows_rows, which need the shape right first."""
    a = call(L)
    if a.shape != (V + N, 16):
        fail("The grown matrix doesn't have the right shape yet. Fix \"extend returns the matrix with one new row per new token\" above first.")
    return a


@test("grows_rows", "extend returns the matrix with one new row per new token")
def _(L):
    a = call(L)
    if a.shape == (N, 16):
        fail(f"`extend` returns only the new rows, shape [{N}, 16]. The model still needs its {V} old rows: put the new ones after them.")
    if a.shape != (V + N, 16):
        fail(f"A [{V}, 16] matrix with {N} new tokens should come back as [{V + N}, 16], but `extend` returns {list(a.shape)}.")


@test("old_rows_kept", "Every old row keeps its place")
def _(L):
    a = grown(L)
    w = arr(W).astype(np.float64)
    if np.allclose(a[:V], w, atol=1e-6):
        return
    if np.allclose(a[N:], w, atol=1e-6):
        fail(f"The new rows come first, so every old id now reads a row {N} places away from its own. Token 0's row is now a new token's. "
             "Old ids keep their numbers, so the old rows have to stay where they are. Put the new rows after them.")
    fail("The first rows of the result aren't the old matrix anymore. `extend` must leave every old row exactly as it was.")


@test("mean_of_pieces", "Each new row is the mean of its pieces' rows")
def _(L):
    a = grown(L)
    w = arr(W).astype(np.float64)
    if not np.allclose(a[:V], w, atol=1e-6):
        fail("The old rows have moved. Fix \"Every old row keeps its place\" above first.")
    new = a[V:]
    want = expected(W, PIECES)[V:]
    if np.allclose(new, want, atol=1e-5):
        return
    if np.allclose(new, np.stack([w[ids].sum(axis=0) for ids in PIECES]), atol=1e-5):
        fail("Each new row is the sum of its pieces, so a token with 3 pieces starts about 3 times too far from the origin, "
             "longer than any row the model has seen. Average them instead.")
    if np.allclose(new, np.stack([w[ids[0]] for ids in PIECES]), atol=1e-5):
        fail("Each new row copies only the first piece's row, so байна, which was ба + й + на, starts out meaning just ба. "
             "Use all of its pieces.")
    if np.allclose(new, np.tile(new[0], (N, 1)), atol=1e-6):
        fail("All your new rows are identical: the mean of the whole matrix, not of each token's pieces. That's a known baseline "
             "(Hewitt 2021), but then every new token starts out the same. Average only the rows of that token's own pieces.")
    for name in ("E", "H"):
        g = L.get(name)
        ga = arr(g) if isinstance(g, torch.Tensor) else None
        if ga is not None and ga.ndim == 2 and ga.shape[1] == 16 and np.allclose(new, np.stack([ga[ids].astype(np.float64).mean(axis=0) for ids in PIECES]), atol=1e-5):
            fail(f"`extend` ignores the `weight` it's given and averages rows of the script's `{name}`. Then `extend(H, PIECES)` "
                 f"would fill the LM head with rows from `{name}`. Use the argument.")
    k = int(np.argmax(np.abs(new - want).max(axis=1)))
    fail(f"Your new rows don't match the mean of their pieces' rows. New token {k} has pieces {PIECES[k]}, so its row should start "
         f"{np.round(want[k][:3], 3).tolist()}…, but yours starts {np.round(new[k][:3], 3).tolist()}…. Are the rows random, or from the wrong pieces?")
