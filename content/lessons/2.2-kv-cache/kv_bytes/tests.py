from harness import test, fail, learner_fn, as_float, script_error, stdout

GIB = 2**30
# (name, config, ctx, bytes per value). Three shapes, so each missing factor leaves its own fingerprint.
LLAMA = ("Llama 3.1 8B at 8k in fp16", {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096}, 8192, 2)
QWEN = ("a 28-layer model with 28 query heads and 4 KV heads, 1000 tokens, 1-byte values",
        {"num_hidden_layers": 28, "num_attention_heads": 28, "num_key_value_heads": 4, "hidden_size": 3584}, 1000, 1)
TINY = ("your 30M model (6 layers, 8 heads, no GQA), 256 tokens in fp32",
        {"num_hidden_layers": 6, "num_attention_heads": 8, "num_key_value_heads": 8, "hidden_size": 512}, 256, 4)


def expected(cfg, ctx, b):
    return 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * (cfg["hidden_size"] // cfg["num_attention_heads"]) * ctx * b


def got(L, probe):
    fn = learner_fn(L, "kv_cache_bytes")
    _name, cfg, ctx, b = probe
    out = fn(dict(cfg), ctx, b)
    if out is Ellipsis:
        fail("`kv_cache_bytes` still returns `...`, the placeholder.")
    if out is None:
        fail("`kv_cache_bytes` returns nothing. Return the number of bytes.")
    v = as_float(out)
    if v is None:
        fail(f"`kv_cache_bytes` should return a number of bytes, but it returns a {type(out).__name__}.")
    return v


def ratio(L, probe):
    return got(L, probe) / expected(*probe[1:])


def near(a, b):
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


@test("returns_bytes", "kv_cache_bytes returns a size in bytes")
def _(L):
    v = got(L, LLAMA)
    if v <= 0:
        fail(f"For {LLAMA[0]} `kv_cache_bytes` returned {v:g}. A cache size has to be positive.")
    if v < 1e6 and near(v * GIB, expected(*LLAMA[1:])) or near(v * GIB, 4 * expected(*LLAMA[1:])):
        fail(f"For {LLAMA[0]} `kv_cache_bytes` returned {v:g}: that's GiB, already divided by 2**30. "
             "Return plain bytes; the print line converts to GiB.")
    if near(v, 8 * expected(*LLAMA[1:])):
        fail(f"For {LLAMA[0]} you got {v / GIB:.2f} GiB, 8× too much: that counts bits. "
             "`bytes_per_value` is already in bytes (fp16 is 2), so don't multiply by 8.")


@test("uses_kv_heads", "Counts the KV heads, not the query heads")
def _(L):
    r1, r2 = ratio(L, LLAMA), ratio(L, QWEN)
    if near(r1, 4) and near(r2, 7):
        fail(f"For {LLAMA[0]} you got {got(L, LLAMA) / GIB:.2f} GiB, 4× too much. That's 4096 values per token per layer for K, "
             "as if every one of the 32 query heads had its own K and V. Llama 3.1 uses grouped-query attention: "
             "4 query heads share each K/V head, so only `num_key_value_heads` = 8 heads of width 128 are cached. "
             "Which config entry counts those? If you already use it, check the head width: 4096 is split across the 32 *query* heads.")


@test("every_factor", "2 × layers × kv_heads × head_dim × ctx × bytes")
def _(L):
    rs = [ratio(L, p) for p in (LLAMA, QWEN, TINY)]
    if all(near(r, 1) for r in rs):
        return
    if all(near(r, 0.5) for r in rs):
        fail(f"For {LLAMA[0]} you got {got(L, LLAMA) / GIB:.2f} GiB, half the size: the cache holds two tensors per layer. What are they?")
    if all(near(r, 1 / p[3]) for r, p in zip(rs, (LLAMA, QWEN, TINY))):
        fail(f"For {LLAMA[0]} you got {got(L, LLAMA) / GIB:.2f} GiB. That counts values, not bytes: "
             "with fp16 every value takes 2 bytes, with fp32 4. Where does `bytes_per_value` go?")
    if all(near(r, 1 / p[1]["num_hidden_layers"]) for r, p in zip(rs, (LLAMA, QWEN, TINY))):
        fail(f"For {LLAMA[0]} you got {got(L, LLAMA) / GIB:.3f} GiB, the size of one layer's cache. Every layer keeps its own K and V.")
    if near(rs[2], 1) and not near(rs[0], 1):
        fail(f"For {TINY[0]} you get the right size, but for {LLAMA[0]} you got {got(L, LLAMA) / GIB:.2f} GiB instead of 1.00. "
             "The tiny model has as many KV heads as query heads. How does the head width depend on that?")
    shown = ", ".join(f"{got(L, p):,.0f}" for p in (LLAMA, QWEN, TINY))
    want = ", ".join(f"{expected(*p[1:]):,}" for p in (LLAMA, QWEN, TINY))
    fail(f"On three configs you got [{shown}] bytes, but they should be [{want}]. "
         "Multiply: 2 (K and V) × layers × KV heads × head width × positions × bytes per value.")


@test("script_runs", "The script prints Llama 3.1 8B's cache size")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "GiB of KV cache" not in stdout():
        fail("The lines that print Llama 3.1 8B's cache size are missing. Put them back at the bottom.")
