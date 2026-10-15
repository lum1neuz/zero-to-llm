from harness import test, fail, learner_fn, as_float, script_error, stdout

B = 1_000_000_000  # one billion weights keeps the expected numbers readable


def gb(L, n_quant, n_bf16, **kw):
    fn = learner_fn(L, "weight_gb")
    out = fn(n_quant, n_bf16, **kw)
    if out is None:
        fail("`weight_gb` returns nothing. Keep the `return total_bytes / 1e9` line at the end.")
    v = as_float(out)
    if v is None:
        fail(f"`weight_gb` should return a number of GB, but it returns {type(out).__name__}.")
    return v


def near(a, b):
    return abs(a - b) <= 1e-9 + 1e-7 * abs(b)


@test("absmax_overhead", "Without double quant: 4 bits plus an fp32 absmax per 64 weights")
def _(L):
    v = gb(L, B, 0, double_quant=False)
    if near(v, 0.5625):
        return
    if near(v, 0.5):
        fail("1B NF4 weights come out at exactly 0.5 GB, so the absmaxes cost nothing. Every 64 weights share one fp32 absmax. "
             "How many bits is that per weight?")
    if near(v, 4.5):
        fail("1B NF4 weights come out at 4.5 GB, 8× too big. 4.5 is the bits per weight. How many bits in a byte?")
    if near(v, 0.53125):
        fail("Your absmax costs 16 bits per block. Without double quantization bitsandbytes keeps each block's absmax in fp32, "
             "32 bits per 64 weights.")
    if near(v, 0.515869140625):
        fail("With `double_quant=False` you still get the double-quantized size. Check the side of `scale_bits` that runs without it.")
    fail(f"1B NF4 weights with one fp32 absmax per 64 should take 0.5625 GB (4.5 bits each). Yours come to {v:.6f} GB.")


@test("double_quant", "With double quant: an 8-bit absmax per 64, plus an fp32 constant per 256 absmaxes")
def _(L):
    v = gb(L, B, 0)
    want = (4 + 8 / 64 + 32 / (64 * 256)) / 8
    if near(v, want):
        return
    if near(v, 0.5625):
        fail("With double quantization on you still pay 4.5 bits per weight. The absmaxes themselves get quantized to 8 bits, "
             "so one costs 8 bits per 64 weights instead of 32. Does `double_quant` change anything in your code?")
    if near(v, 0.515625):
        fail("Close. You left out the fp32 constants that the 8-bit absmaxes are scaled by, one per 256 blocks. "
             "It's only 32 / (64 × 256) ≈ 0.002 bits per weight, but without it the 8-bit absmaxes can't be read back.")
    if near(v, (4 + 32 / (64 * 256)) / 8):
        fail("You counted only the fp32 constants. The 8-bit absmaxes themselves still cost 8 bits per 64 weights.")
    if near(v, want * 8):
        fail(f"1B weights come out at {v:.3f} GB, 8× too big. That's bits per weight, not bytes.")
    fail(f"1B NF4 weights with double quantization should take {want:.6f} GB ({want * 8:.4f} bits each). Yours come to {v:.6f} GB.")


@test("bf16_part", "embed_tokens and lm_head stay bf16, 2 bytes each")
def _(L):
    v = gb(L, 0, B)
    if near(v, 2.0):
        return
    if near(v, 0.0):
        fail("1B bf16 weights cost 0 GB in your sum. `n_bf16` is the embedding and the output head, 1.05B weights that "
             "bitsandbytes leaves unquantized. Add them in.")
    if near(v, 4.0):
        fail("You count 4 bytes per bf16 weight. That's fp32. bf16 is 16 bits.")
    if near(v, 1.0):
        fail("You count 1 byte per bf16 weight. bf16 is 16 bits, 2 bytes.")
    if near(v, 16.0):
        fail("1B bf16 weights come out at 16 GB. That's bits, not bytes.")
    if near(v, 0.5625) or near(v, 0.515869140625):
        fail("You quantized the bf16 weights too. bitsandbytes only converts the Linear layers inside the blocks; "
             "`embed_tokens` and `lm_head` stay bf16.")
    fail(f"1B bf16 weights should take 2.0 GB. Yours come to {v:.4f} GB.")


@test("script_runs", "The script prints all three sizes")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if stdout().count("GB") < 3:
        fail("The three print lines at the bottom are missing. Put them back to compare NF4 with bf16.")
