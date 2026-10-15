from harness import test, fail, learner_fn, script_error, stdout

B = 4
SHARED = [1, 2, 3, 4, 5, 6, 7, 8]
X = SHARED + [9, 9]                  # 2 full blocks, then 2 tokens that don't fill a block
Y = SHARED + [5, 5, 5, 5, 6]         # same 2 blocks, then a different full block
P = [1, 2, 3, 4, 5, 6, 7, 8]
Q = [9, 9, 9, 9, 5, 6, 7, 8]         # block 1 has P's tokens, after a different block 0


def hashes(L, tokens):
    learner_fn(L, "block_hashes")   # explains a missing or renamed function
    try:
        out = L["block_hashes"](list(tokens), B)
    except TypeError as e:
        if "unhashable" in str(e):
            fail(f"`block_hashes` raised TypeError: {e}. A list can change, so Python won't hash it. "
                 "Turn the block into a tuple first: `tuple(tokens[...])`.")
        out = learner_fn(L, "block_hashes")(list(tokens), B)
    except Exception:  # noqa: BLE001 - let the harness explain it
        out = learner_fn(L, "block_hashes")(list(tokens), B)
    if not isinstance(out, list):
        fail(f"`block_hashes` should return a list of hashes, but it returns {type(out).__name__}. Keep `return hashes`.")
    return out


def full(L, tokens):
    """hashes() for the tests after full_blocks, which need one hash per full block first."""
    out = hashes(L, tokens)
    if len(out) != len(tokens) // B:
        fail("`block_hashes` doesn't return one hash per full block yet. Fix \"One hash per full block\" above first.")
    return out


@test("full_blocks", "One hash per full block")
def _(L):
    h = hashes(L, X)
    if not h:
        fail("`block_hashes` returns no hashes. Fill in the loop that appends one hash per full block.")
    if len(h) == 3:
        fail(f"{len(X)} tokens in blocks of {B} make 2 full blocks, but you return 3 hashes. The last 2 tokens are a block "
             "still being filled: its KV changes as the request grows, so it can't be shared yet. Hash full blocks only.")
    if len(h) != len(X) // B:
        fail(f"{len(X)} tokens in blocks of {B} make {len(X) // B} full blocks, but you return {len(h)} hashes.")


@test("shared_prefix", "Prompts with the same start get the same hashes for it")
def _(L):
    hx, hy = full(L, X), full(L, Y)
    if hx != full(L, X):
        fail("Hashing the same tokens twice gives different hashes. A hash must depend only on the tokens.")
    if hx[:2] != hy[:2]:
        fail("Two prompts that start with the same 8 tokens get different hashes for those first 2 blocks, so nothing would "
             "ever be reused. A block's hash should cover the tokens up to the end of that block, nothing after it.")


@test("chained", "The same block after a different start gets a different hash")
def _(L):
    hp, hq = full(L, P), full(L, Q)
    if hp[1] == hq[1]:
        fail("Block 1 is [5, 6, 7, 8] in both prompts, but the tokens before it differ, so its keys and values differ too: "
             "attention mixed in a different past. Your hashes match, so one request would reuse the other's KV. "
             "Each block's hash has to include its parent's hash.")


@test("prefix_len", "cached_prefix_len counts tokens, up to the first miss")
def _(L):
    fn = learner_fn(L, "cached_prefix_len")
    tokens = SHARED + [5, 5, 5, 5, 6, 6]   # 3 full blocks + 2 tokens
    h = full(L, tokens)
    got = fn(list(tokens), set(h), B)
    if got is None:
        fail("`cached_prefix_len` returns nothing. It has to `return` the number of cached tokens.")
    if got is Ellipsis:
        fail("`cached_prefix_len` still returns `...`, the placeholder.")
    if got == 3:
        fail("Every block is cached and you return 3: that counts blocks. The caller needs tokens, block_size per block.")
    if got == len(tokens):
        fail(f"You return {got}, every token. The last 2 don't fill a block, so they were never cached. Count whole blocks only.")
    if got != 12:
        fail(f"All 3 full blocks are cached, so 12 tokens can be reused, but you return {got}.")
    evicted = set(h) - {h[1]}   # block 1 was evicted; block 2's entry is still there
    got = fn(list(tokens), evicted, B)
    if got == 8:
        fail("With block 1 evicted you return 8: you counted block 2 after the miss. Block 2's KV was computed on top of "
             "block 1's, so without block 1 it can't be used. Stop at the first miss.")
    if got != 4:
        fail(f"With block 1 evicted, only block 0 (4 tokens) is reusable, but you return {got}.")


@test("script_runs", "The script runs and prints the reuse")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "hit" not in L:
        fail("The lines at the bottom that fill the cache and look up the second request are missing. Put them back.")
    if not stdout().strip():
        fail("Nothing printed. Put the print lines back.")
