from harness import test, fail, learner_fn, script_error, stdout


def counts(L):
    """block_params at d = 1…4, checked to be whole numbers."""
    fn = learner_fn(L, "block_params")
    out = []
    for d in (1, 2, 3, 4):
        n = fn(d)
        if n is None:
            fail("`block_params` returns nothing. It has to end with `return attn + mlp + norms`.")
        if n is Ellipsis:
            fail("`block_params` returns `...`, the placeholder.")
        if not isinstance(n, (int, float)) or isinstance(n, bool):
            fail(f"`block_params` should return a number, but it returns {type(n).__name__}.")
        if isinstance(n, float) and not n.is_integer():
            fail(f"`block_params({d})` is {n}. A parameter count is a whole number, so something divides where it should multiply.")
        out.append(int(n))
    return out


def poly(L):
    """Read the learner's formula as a·d² + b·d + c from four values."""
    f1, f2, f3, f4 = counts(L)
    a2 = f3 - 2 * f2 + f1           # 2a
    a = a2 / 2
    b = f2 - f1 - 3 * a
    c = f1 - a - b
    if 16 * a + 4 * b + c != f4:
        fail("Your count doesn't grow like a·d² + b·d. Every weight matrix in a block is some multiple of d × d, "
             "and every bias or LayerNorm vector some multiple of d.")
    return a, b, c


@test("returns_number", "block_params returns a whole number")
def _(L):
    counts(L)


@test("weights", "The four Linear layers hold 12·d² weights")
def _(L):
    a, _b, _c = poly(L)
    if a == 12:
        return
    stories = {
        6: "The MLP widens each token to 4·d features and back: c_fc is d × 4d and its c_proj is 4d × d, 8·d² together. Yours has 2·d².",
        10: "c_attn makes q, k *and* v in one matmul, so it maps d features to 3d: 3·d² weights, not d².",
        11: "Attention has two Linear layers: c_attn (d × 3d) and its output projection c_proj (d × d). One of them is missing.",
        8: "Your matrices add up to 8·d², the MLP alone. Attention's c_attn and c_proj hold 4·d² more.",
        4: "Your matrices add up to 4·d², attention alone. The MLP's c_fc and c_proj hold 8·d² more.",
    }
    fail(stories.get(a, f"Your formula has {a:g}·d² weights. The four matrices are d × 3d, d × d, d × 4d and 4d × d: 12·d² in all."))


@test("biases_and_norms", "Biases and LayerNorms add 13·d")
def _(L):
    _a, b, c = poly(L)
    if c != 0:
        fail(f"Your formula has a constant term of {c:g}. Every parameter in a block grows with d, so there's nothing to add on its own.")
    if b == 13:
        return
    stories = {
        9: "You've counted the biases but not the LayerNorms. Each LayerNorm learns a gain and a bias per feature: 2 × 2·d = 4·d.",
        11: "You're 2·d short. Check two things: each LayerNorm learns a gain *and* a bias (4·d for two of them), "
            "and each Linear has one bias per *output* (c_attn 3d, c_proj d, c_fc 4d, c_proj d).",
        4: "The Linear biases are missing. Each nn.Linear has one bias per *output*: c_attn 3d, c_proj d, c_fc 4d, c_proj d.",
        8: "Each nn.Linear has one bias per *output*, not per input: c_attn 3d, c_proj d, c_fc 4d and c_proj d, 9·d in all.",
        10: "You're 3·d short. Check the biases of the layers whose output isn't d wide: c_attn has 3d outputs and c_fc 4d.",
    }
    fail(stories.get(b, f"Your formula has {b:g}·d in biases and LayerNorms. The biases are 3d + d + 4d + d = 9·d and the two LayerNorms 4·d."))


@test("script_runs", "The script compares your formula with a built Block")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "d=512" not in stdout():
        fail("The comparison loop didn't print. Put it back at the bottom so you can see d = 512.")
