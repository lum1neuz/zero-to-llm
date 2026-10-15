from harness import test, fail, warn, learner_fn, as_float, script_error, stdout

# Two small "Linear layers", not square, so in and out can't be swapped by accident.
SMALL = [(4, 6), (6, 3)]
LLAMA = [(4096, 4096), (4096, 1024), (4096, 1024), (4096, 4096), (4096, 14336), (4096, 14336), (14336, 4096)]


def near(a, b):
    return abs(a - b) <= 1e-9 + 1e-7 * abs(b)


def number(L, name, *args, **kw):
    fn = learner_fn(L, name)
    out = fn(*args, **kw)
    if out is Ellipsis:
        fail(f"`{name}` still returns `...`, the placeholder.")
    if out is None:
        fail(f"`{name}` returns nothing. It needs a `return`.")
    v = as_float(out)
    if v is None:
        fail(f"`{name}` should return a number, but it returns {type(out).__name__}.")
    return v


def check_step1(L):
    fn = L.get("weight_gb")
    try:
        ok = callable(fn) and near(float(fn(10 ** 9, 0)), (4 + 8 / 64 + 32 / (64 * 256)) / 8)
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        warn("note: weight_gb no longer passes step 1's tests")


@test("lora_counts", "lora_params counts A and B in every adapted layer")
def _(L):
    check_step1(L)
    v = number(L, "lora_params", SMALL, r=2, layers=3)
    if near(v, 114):
        v = number(L, "lora_params", LLAMA, r=16, layers=32)
        if not near(v, 41_943_040):
            fail(f"Your count is right on small layers but gives {v:,.0f} for Llama 3.1 8B instead of 41,943,040. "
                 "Does it use its arguments rather than the script's globals?")
        return
    if near(v, 41_943_040):
        fail("On two small layers your count is 41,943,040, Llama 3.1 8B's number. Does it use its arguments, or the script's `SHAPES`, `LAYERS` and r = 16?")
    if near(v, 126):
        fail("Two layers [4 → 6] and [6 → 3], r = 2, 3 blocks: you get 126, which is in × out, the full weights. "
             "LoRA trains A ([r, in]) and B ([out, r]) instead. How many numbers are in those two?")
    if near(v, 252):
        fail("You get r × in × out per layer. A is [r, in] and B is [out, r], two separate thin matrices, so their sizes add.")
    if near(v, 38):
        fail("You get 38, the count for one block. The adapters sit in every block. Multiply by `layers`.")
    if near(v, 60):
        fail("You count only A (r × in). B adds out × r per layer too.")
    if near(v, 54):
        fail("You count only B (out × r). A adds r × in per layer too.")
    fail(f"Two layers [4 → 6] and [6 → 3], r = 2, 3 blocks should give 3 × 2 × (10 + 9) = 114. Yours gives {v:,.0f}.")


@test("adapter_state", "adapter_gb counts 16 bytes per trainable parameter")
def _(L):
    v = number(L, "adapter_gb", 10 ** 9)
    if near(v, 16.0):
        return
    stories = {
        2.0: "You count 2 bytes per parameter, only a bf16 copy of the weights. Training also needs the gradient and Adam's two moments, all in fp32 here.",
        4.0: "You count only the fp32 weights. Training also keeps a gradient and Adam's two moments for every trainable parameter.",
        8.0: "You count the weight and its gradient, but not Adam's two moments (m and v), 4 bytes each.",
        12.0: "You count one of Adam's moments. It keeps two per parameter, the running mean (m) and the running square (v).",
        128.0: "1B trainable parameters come out at 128 GB. That's bits, not bytes.",
    }
    for want, story in stories.items():
        if near(v, want):
            fail(story)
    fail(f"1B trainable parameters should cost 16 GB (4 bytes each for weight, gradient, m and v). Yours cost {v:.3f} GB.")


@test("logits_size", "logits_gb is seq × vocab × batch fp32 numbers")
def _(L):
    v = number(L, "logits_gb", 1000, 50_000, batch=3)
    if near(v, 0.6):
        return
    if near(v, 0.3):
        fail("You count 2 bytes per logit. The model runs in bf16, but the loss upcasts the logits to fp32 first, 4 bytes each.")
    if near(v, 0.2):
        fail("With a batch of 3 your logits come out at the size of one sequence. Every sequence in the batch has its own.")
    if near(v, 0.1):
        fail("You count 2 bytes per logit and one sequence. The loss upcasts the logits to fp32 (4 bytes), and every sequence in the batch has its own.")
    if near(v, 4.8):
        fail("Your logits come out at 4.8 GB, 8× too big. That's bits, not bytes.")
    if near(v, 1000 * 4096 * 3 * 4 / 1e9):
        fail("You used the hidden size. The output head turns each token's 4096 hidden numbers into one score per vocabulary entry, so it's `vocab` wide.")
    fail(f"1000 tokens × 50,000 entries × batch 3, at 4 bytes each, should be 0.6 GB. Yours is {v:.4f} GB.")


@test("script_runs", "The budget prints, and fits in 12 GB")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    total = as_float(L.get("total"))
    if total is None or "total" not in stdout():
        fail("The budget loop at the bottom is missing. Put it back to see the total.")
    if not 9.5 < total < 10.5:
        fail(f"Your total is {total:.2f} GB, but with the parts above it should be about 10.1 GB. Did a line of the budget change?")
