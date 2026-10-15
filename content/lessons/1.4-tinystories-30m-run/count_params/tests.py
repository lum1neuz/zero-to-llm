from harness import test, fail, learner_fn

BASE = dict(vocab_size=21504, block_size=512, n_layer=6, n_head=8, n_embd=512)
D = 512
BLOCK = 12 * D * D + 13 * D


def count(L, **change):
    fn = learner_fn(L, "count_params")
    cfg = dict(BASE, **change)
    n = fn(cfg)
    if n is Ellipsis:
        fail("`count_params` returns `...`, a placeholder.")
    if n is None:
        fail("`count_params` returns nothing. It has to end with `return embeddings + blocks + final`.")
    if not isinstance(n, (int, float)) or isinstance(n, bool):
        fail(f"`count_params` should return a number, but it returns {type(n).__name__}.")
    if isinstance(n, float) and not n.is_integer():
        fail(f"`count_params` returns {n}. A parameter count is a whole number, so something divides where it should multiply.")
    return int(n)


def fmt(n):
    return f"{n:,}".replace("-", "−")


@test("returns_number", "count_params returns a whole number")
def _(L):
    count(L)


@test("embeddings", "Counts the token embedding once (lm_head is tied) and one vector per position")
def _(L):
    per_token = count(L, vocab_size=21505) - count(L)
    if per_token == 2 * D:
        fail(f"One more vocab entry adds {fmt(per_token)} parameters: you counted the token embedding and lm_head separately. "
             "With weight tying, lm_head *is* wte, so each vocab entry costs one d-vector.")
    if per_token == 2 * D + 1:
        fail(f"One more vocab entry adds {fmt(per_token)} parameters: a separate lm_head with a bias. "
             "With weight tying, lm_head reuses wte and has no bias, so each vocab entry costs one d-vector.")
    if per_token == 0:
        fail("The vocab size doesn't change your count, so the token embedding is missing. wte holds one d-vector per vocab entry.")
    if per_token != D:
        fail(f"One more vocab entry adds {fmt(per_token)} parameters. It should add one d-vector, {D}.")
    per_pos = count(L, block_size=513) - count(L)
    if per_pos == 0:
        fail("block_size doesn't change your count, so the position embedding is missing. wpe holds one d-vector per position.")
    if per_pos != D:
        fail(f"One more position adds {fmt(per_pos)} parameters. It should add one d-vector, {D}.")


@test("blocks", "Counts n_layer blocks of 12·d² + 13·d each")
def _(L):
    if count(L, n_head=4) != count(L):
        fail("Changing n_head changes your count, but heads only cut c_attn's output into slices. They add no parameters.")
    per_block = count(L, n_layer=7) - count(L)
    if per_block == BLOCK:
        return
    if per_block == 0:
        fail("n_layer doesn't change your count, so you've counted one block. Multiply by the number of blocks.")
    if per_block == 12 * D * D:
        fail(f"Each block adds {fmt(per_block)}, only its matrices. Add the biases and the two LayerNorms, 13·d per block.")
    fail(f"Each extra block adds {fmt(per_block)} parameters to your count. 1.2's block_params gives 12·d² + 13·d = {fmt(BLOCK)}.")


@test("total", "The 30M config has 30,187,520 parameters")
def _(L):
    n = count(L)
    want = 21504 * D + 512 * D + 6 * BLOCK + 2 * D
    if n == want:
        return
    off = n - want
    if off == -2 * D:
        fail("You're 1,024 short: the final LayerNorm, ln_f, between the last block and lm_head. It has a gain and a bias.")
    if off == -D:
        fail("You're 512 short. The final LayerNorm learns a gain *and* a bias, d values each.")
    fail(f"Your count is {fmt(n)}, off by {fmt(off)}. The embeddings and blocks check out, so look at `final`.")
