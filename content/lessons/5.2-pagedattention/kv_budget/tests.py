from harness import test, fail, learner_fn, as_float, script_error, stdout

LLAMA_8B = {"layers": 32, "heads": 32, "kv_heads": 8, "head_dim": 128}
LLAMA_70B = {"layers": 80, "heads": 64, "kv_heads": 8, "head_dim": 128}
# Five requests, different from the script's. 16 tokens fill exactly one block; 31 needs two.
LENGTHS = [100, 250, 31, 700, 16]
MAX_LEN = 1024
BLOCK = 16
USED = sum(LENGTHS)


def kv(cfg, b):
    return 2 * cfg["layers"] * cfg["kv_heads"] * cfg["head_dim"] * b


def kv_value(L, cfg, b):
    fn = learner_fn(L, "kv_bytes_per_token")
    out = fn(dict(cfg), b)
    if out is None:
        fail("`kv_bytes_per_token` returns nothing. It has to `return` the bytes per token.")
    if out is Ellipsis:
        fail("`kv_bytes_per_token` still returns `...`, the placeholder.")
    v = as_float(out)
    if v is None:
        fail(f"`kv_bytes_per_token` should return a number of bytes, but it returns {type(out).__name__}.")
    return v


@test("kv_per_token", "KV bytes per token use kv_heads, K and V, every layer")
def _(L):
    v = kv_value(L, LLAMA_8B, 2)
    right = kv(LLAMA_8B, 2)
    if v == right:
        v70 = kv_value(L, LLAMA_70B, 1)
        if v70 != kv(LLAMA_70B, 1):
            fail(f"Right for the 8B model, but for a 70B config (80 layers, 8 KV heads, int8 values) you get {v70:.0f} bytes "
                 f"instead of {kv(LLAMA_70B, 1)}. Read every number from `cfg` and `bytes_per_value` instead of typing 8B's.")
        return
    kib = v / 1024
    if v == right * cfg_ratio("heads"):
        fail(f"You get {kib:.0f} KiB per token, 4× too much: you counted `heads` (32). Llama 3.1 8B has 32 *query* heads "
             "but only 8 KV heads. With grouped-query attention, 4 query heads share each K and V, and the cache stores K and V. "
             "Count `kv_heads`.")
    if v == right / 32:
        fail(f"You get {kib:.0f} KiB per token, one layer's worth. Every one of the 32 layers keeps its own K and V.")
    if v == right / 2:
        v4 = kv_value(L, LLAMA_8B, 4)
        if v4 == kv(LLAMA_8B, 4) / 4:
            fail(f"You get {kib:.0f} KiB per token: that counts values, not bytes. Each value takes `bytes_per_value` bytes "
                 "(2 in fp16, 4 in fp32).")
        fail(f"You get {kib:.0f} KiB per token, half of 128. Every layer caches two vectors per KV head and token: a key *and* a value.")
    fail(f"You get {v:.0f} bytes per token. Expected 2 (K and V) × 32 layers × 8 KV heads × 128 dims × 2 bytes = {right} bytes, 128 KiB.")


def cfg_ratio(key):
    return LLAMA_8B[key] / LLAMA_8B["kv_heads"]


def waste(L):
    fn = learner_fn(L, "waste_pct")
    out = fn(list(LENGTHS), MAX_LEN, BLOCK)
    if not isinstance(out, tuple) or len(out) != 2:
        fail("`waste_pct` should return two percentages, contiguous then paged. Keep its `return` line.")
    vals = [as_float(x) for x in out]
    if None in vals:
        fail("`waste_pct` should return two numbers.")
    return vals


def reserved_from(pct):
    """The slot count a waste percentage implies, from 100 × (1 − used / reserved)."""
    return USED / (1 - pct / 100) if pct != 100 else float("inf")


@test("contiguous_waste", "Contiguous reserves max_len for every request")
def _(L):
    w = waste(L)[0]
    right = len(LENGTHS) * MAX_LEN
    got = reserved_from(w)
    if abs(got - right) < 0.5:
        return
    if abs(got - MAX_LEN) < 0.5:
        fail(f"Your contiguous waste is {w:.1f}%: you reserved {MAX_LEN} slots in total for 5 requests holding {USED} tokens. "
             "Each request reserves its own `max_len` up front.")
    if abs(w) < 1e-9:
        fail("Your contiguous waste is 0%, as if each request reserved only what it uses. A contiguous server can't know how long "
             "an answer will grow, so it reserves `max_len` slots for every request when it arrives.")
    fail(f"Your contiguous waste is {w:.1f}%. 5 requests × {MAX_LEN} slots = {right} reserved for {USED} tokens is "
         f"{100 * (1 - USED / right):.1f}%.")


@test("paged_waste", "Paged rounds each request up to whole blocks")
def _(L):
    w = waste(L)[1]
    right = sum(-(-n // BLOCK) * BLOCK for n in LENGTHS)
    got = reserved_from(w)
    if abs(got - right) < 0.5:
        return
    if abs(got - sum(n // BLOCK * BLOCK for n in LENGTHS)) < 0.5:
        fail(f"Your paged waste is {w:.1f}%: negative, because you reserve fewer slots ({got:.0f}) than the {USED} tokens. "
             "A 31-token request gets one block of 16 and loses 15 tokens: `31 // 16` rounds down. How do you round up?")
    if abs(got - sum((n // BLOCK + 1) * BLOCK for n in LENGTHS)) < 0.5:
        fail(f"Your paged waste is {w:.1f}%: you add a block even when the last one is full. The 16-token request fills exactly "
             "one block, but you reserve two. Round up only when there's a remainder.")
    if abs(got - (-(-USED // BLOCK)) * BLOCK) < 0.5:
        fail(f"Your paged waste is {w:.1f}%: you rounded up the *total*, as if requests shared blocks. Each block belongs to one "
             "request (its block table points at it), so each request rounds up on its own.")
    if abs(got - sum(-(-n // BLOCK) for n in LENGTHS)) < 0.5:
        fail(f"Your paged waste is {w:.1f}%: you counted blocks ({got:.0f}), not slots. Each block holds `block_size` tokens.")
    if abs(w) < 1e-9:
        fail("Your paged waste is 0%, as if every request filled its last block exactly. The 100-token request needs 7 blocks "
             "of 16 = 112 slots, so 12 sit empty. Reserve whole blocks.")
    fail(f"Your paged waste is {w:.2f}%. Rounding each request up to whole blocks of 16 reserves {right} slots for {USED} tokens: "
         f"{100 * (1 - USED / right):.2f}%.")


@test("script_runs", "The script runs and prints both budgets")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "paged" not in L:
        fail("The lines at the bottom that call your functions and print are missing. Put them back.")
    if not stdout().strip():
        fail("Nothing printed. Put the print lines back.")
