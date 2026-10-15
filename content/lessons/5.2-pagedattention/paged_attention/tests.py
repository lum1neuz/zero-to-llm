import numpy as np
import torch
from harness import test, fail, warn, learner_fn, arr, script_error, stdout

D = 8
N_BLOCKS = 10
# (block table, length): a half-full last block, an exactly full block, and a 1-token request.
REQUESTS = [([5, 2, 9], 10), ([7], 4), ([1, 3, 8, 0], 13), ([6], 1)]


def block(L):
    b = L.get("BLOCK")
    return b if isinstance(b, int) and b > 0 else 4


def pools(B):
    """Every slot holds something: stale keys and values from earlier requests, like a real pool."""
    g = np.random.default_rng(7)
    return (g.standard_normal((N_BLOCKS, B, D)).astype(np.float32),
            g.standard_normal((N_BLOCKS, B, D)).astype(np.float32),
            g.standard_normal(D).astype(np.float32))


def attend(q, k, v, scale):
    s = k.astype(np.float64) @ q.astype(np.float64) * scale
    w = np.exp(s - s.max())
    return (w / w.sum()) @ v.astype(np.float64)


def gathered(P, table, n):
    return P[table].reshape(-1, D)[:n]


def check_step1(L):
    """Warn (don't fail) when step 1's gather_kv has drifted since it was solved."""
    fn = L.get("gather_kv")
    if not callable(fn):
        return
    B = block(L)
    P = np.arange(N_BLOCKS * B * 3, dtype=np.float32).reshape(N_BLOCKS, B, 3)
    try:
        a = arr(fn(torch.tensor(P), [3, 1], B + 1))
        ok = a is not None and np.array_equal(a, P[[3, 1]].reshape(-1, 3)[:B + 1])
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: gather_kv no longer passes step 1's tests")


@test("returns_vector", "paged_attention returns one output vector of size d")
def _(L):
    check_step1(L)
    B = block(L)
    K, V, q = pools(B)
    table, n = REQUESTS[0]
    out = learner_fn(L, "paged_attention")(torch.tensor(q), torch.tensor(K), torch.tensor(V), list(table), n)
    if out is None:
        fail("`paged_attention` returns nothing. Keep `return weights @ v`.")
    if out is Ellipsis:
        fail("`paged_attention` returns `...`. Keep `return weights @ v` and fill in the placeholders above it.")
    a = arr(out)
    if a is None:
        fail(f"`paged_attention` should return a tensor, but it returns {type(out).__name__}.")
    if a.shape != (D,):
        fail(f"`paged_attention` returns shape [{', '.join(map(str, a.shape))}], but one query with d = {D} should give [{D}]: "
             "a weighted mix of the value rows. Check that `weights` has one entry per cached token.")


@test("matches_dense", "Paged attention matches dense attention on the same tokens")
def _(L):
    B = block(L)
    K, V, q = pools(B)
    fn = learner_fn(L, "paged_attention")
    for table, n in REQUESTS:
        out = arr(fn(torch.tensor(q), torch.tensor(K), torch.tensor(V), list(table), n))
        if out is None or out.shape != (D,):
            fail(f"For block table {table} with {n} tokens, `paged_attention` doesn't return a [{D}] vector. Fix the test above first.")
        k, v = gathered(K, table, n), gathered(V, table, n)
        right = attend(q, k, v, D ** -0.5)
        if np.allclose(out, right, atol=1e-5):
            continue
        full = len(table) * B
        if full != n and np.allclose(out, attend(q, K[table].reshape(-1, D), V[table].reshape(-1, D), D ** -0.5), atol=1e-5):
            fail(f"For block table {table} holding {n} tokens, your output also attends to the {full - n} slots after the last token. "
                 "They hold stale keys and values from an earlier request. Gather only `length` tokens.")
        if np.allclose(out, attend(q, k, v, 1.0), atol=1e-5):
            fail("Your output matches attention without the 1/√d scale. Without it, q·k grows with d and the softmax gets too peaky. "
                 "Divide the scores by √d, where d is the key size.")
        if np.allclose(out, attend(q, k, v, 1.0 / D), atol=1e-5):
            fail(f"You divide the scores by d ({D}), not √d ({D ** 0.5:.2f}). The scale is 1/√d.")
        if np.allclose(out, attend(q, k, k, D ** -0.5), atol=1e-5):
            fail("Your output mixes the *keys*, not the values. `v` should come from `v_pool`.")
        if np.allclose(out, (k.astype(np.float64) @ q * D ** -0.5) @ v, atol=1e-4):
            fail("Your weights are raw scores, so they don't sum to 1. Softmax turns q·k / √d into weights over the cached tokens.")
        fail(f"For block table {table} with {n} tokens, your output is off by up to {np.abs(out - right).max():.3g} from dense attention "
             "over the same tokens. Gather k and v with your `gather_kv`, then softmax(k @ q / √d) weights the rows of v.")


@test("script_runs", "The script runs and prints the comparison")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "paged" not in L:
        fail("The lines at the bottom that build `dense` and `paged` and print the difference are missing. Put them back.")
    if not stdout().strip():
        fail("Nothing printed. Put the print line back.")
