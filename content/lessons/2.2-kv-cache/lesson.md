---
id: "2.2"
title: KV cache
summary: Keep every token's key and value so each step runs one new token. Match 2.1's logits exactly, then size the cache for a real model with kv_heads.
phase: "2"
hours: 5
type: build
status: ready
protected: false
cut: null
prerequisites:
  - "2.1"
papers:
  - title: Efficiently Scaling Transformer Inference
    authors: Pope et al.
    year: 2022
    url: https://arxiv.org/abs/2211.05102
  - title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
    authors: Ainslie et al.
    year: 2023
    url: https://arxiv.org/abs/2305.13245
quality_metric:
  id: logits_match
  label: Cached logits match the no-cache loop
  unit: max abs diff
  inputs:
    - { id: max_diff, label: "Max |cached − no-cache| logit difference", hint: "fp32, 512 tokens, printed by --compare-cache" }
  compute: value
  reference: 0.0000019
  pass: { max: 0.0001 }
systems_metric:
  id: kv_bytes_per_seq
  label: KV cache size per sequence
  unit: GiB
  note: "2 × layers × kv_heads × head_dim × ctx × bytes. Use kv_heads, not heads. Llama 3.1 8B at 8k: 2 × 32 × 8 × 128 × 8192 × 2 bytes = 1 GiB."
  inputs:
    - { id: kv_gib, label: "Llama 3.1 8B, 8k context, fp16 (GiB)", hint: From your kv_cache_bytes }
  compute: value
  reference: 1.0
  pass: { max: 1.01 }
quiz:
  - id: no_mask
    q: The cached step attends from the new token to every row of the cache, with no causal mask. Why is that safe?
    answer: 0
    options:
      - text: The cache only holds this token and the ones before it, so there's no future to hide
        why: In the full forward, the mask hides positions after each query. Here the query is the newest token, so nothing after it exists yet.
      - text: softmax ignores future tokens on its own
        why: softmax weighs every score it's given. The mask is what removes future positions, and in the cached step there are none.
      - text: It isn't safe. The logits only match approximately without a mask
        why: Your first challenge matched the masked full forward exactly, with no mask in the cached step.
  - id: gqa_cache
    q: Llama 3.1 8B has 32 query heads and 8 KV heads. Compared with a model that gives every query head its own K and V, its KV cache is…
    answer: 1
    options:
      - text: The same size. KV heads only save compute
        why: The cache stores one K and one V per *KV* head, per layer, per position. Fewer KV heads means fewer cached vectors.
      - text: 4× smaller
        why: 32 / 8 = 4 query heads share each K/V head, so the cache holds 8 heads' worth instead of 32. At 8k context that's 1 GiB instead of 4.
      - text: 8× smaller
        why: The saving is query heads ÷ KV heads, 32 / 8 = 4. The 8 is how many KV heads there are, not the ratio.
  - id: why_not_q
    q: Why does the cache keep K and V, but not Q?
    answer: 2
    options:
      - text: Q can be recomputed from K
        why: Q, K and V are three separate projections of x. None of them can be recovered from the others.
      - text: Q is too big to store
        why: q is the same width as k. Storing it would cost the same and buy nothing.
      - text: An old token's query was only used for its own output, but its key and value are read by every later token
        why: Each step needs only the new token's q, plus every earlier k and v. Old queries are never read again.
---

## Goal

Make generation cheap. In a causal model, a token's key and value never change once they're computed, so **cache** them: each step runs only the newest token and attends over the cache. Check that the cached logits match 2.1's no-cache loop, then work out what the cache costs in memory for a real model. Use **kv_heads**, not heads.

## Learn

```card
type: concept
id: why_cache
eyebrow: Why cache
title: Old keys and values never change. Keep them.
visual:
  kind: bars
  max: 2100
  bars:
    - { label: no cache, value: 2010, text: "2,010" }
    - { label: KV cache, value: 64, text: "64" }
body: |
  Token t's key and value come only from tokens 0…t, so they're the same at every later step. 2.1's loop recomputes them anyway. A **KV cache** stores each token's `k` and `v` once. Every new step computes `q`, `k` and `v` for the newest token only and attends over the cache. The bars show positions processed to write 60 characters from a 4-character prompt.
```

```card
type: predict
id: cache_rows
eyebrow: Predict the output
title: After the prompt and 3 new tokens, what's the cache's shape?
code: |
  cache = {"k": torch.zeros(0, 16), "v": torch.zeros(0, 16)}
  for pos in range(len(prompt)):          # prompt has 4 tokens
      logits = step(model, prompt[pos], pos, cache)
  # ... then 3 new tokens, each through step()
  print(cache["k"].shape)
options:
  - text: torch.Size([7, 16])
    correct: true
    why: One row per token that has gone through step(), 4 from the prompt and 3 new ones, each 16 channels wide.
  - text: torch.Size([1, 16])
    why: That's a cache that gets replaced each step instead of growing. The new token would forget everything before it.
  - text: torch.Size([16, 7])
    why: Rows are tokens and columns are channels, so new tokens add rows. `cache["k"] @ q` relies on that layout.
  - text: torch.Size([3, 16])
    why: The prompt goes through the cache too. That's the prefill, and the new tokens need its keys and values.
hint: Every call to step() adds exactly one row. Count the calls.
```

```card
type: fill_blank
id: kv_formula
eyebrow: Fill the blanks
title: Complete the KV cache size.
code: |
  kv_bytes = 2 * layers * [[0]] * head_dim * ctx * [[1]]
tokens: [n_heads, kv_heads, hidden_size, bytes_per_value, "8", vocab_size]
answer: [kv_heads, bytes_per_value]
tips:
  n_heads: Only K/V heads are cached. With grouped-query attention several query heads share one K/V head, so counting query heads overstates the cache, 4× for Llama 3.1 8B.
  hidden_size: hidden_size is already n_heads × head_dim. With head_dim in the product too, you'd count the width twice.
  "8": This counts bytes, not bits. fp16 is 2 bytes per value.
  vocab_size: The vocabulary sizes the embedding and the logits, not the cache.
why_ok: "2 for K and V, then one vector of head_dim values per layer, per KV head, per position, each value taking bytes_per_value bytes."
why_wrong: "Check each blank: which heads actually store a K and a V, and what turns a count of values into bytes?"
hint: The cache stores one K and one V for every head that *has* its own K and V. That's not always every attention head.
```

```card
type: calc
id: llama_kv_gib
eyebrow: Do the math
title: How much KV cache does one 8k-token sequence need on Llama 3.1 8B?
given:
  - { label: Layers, value: "32" }
  - { label: Query heads / KV heads, value: 32 / 8 }
  - { label: head_dim, value: "128" }
  - { label: Context, value: "8192 tokens" }
  - { label: Each value, value: fp16 (2 bytes) }
unit: GiB
answer: { min: 0.99, max: 1.01 }
placeholder: "0.0"
why_ok: 2 × 32 × 8 × 128 × 8192 × 2 = 1,073,741,824 bytes, exactly 1 GiB per sequence. A batch of 16 at 8k needs 16 GiB of cache, about as much as the fp16 weights.
why_wrong: "If you got 4, you used all 32 query heads. Llama 3.1 shares each K/V head between 4 query heads, so only 8 are cached: 2 × 32 × 8 × 128 × 8192 × 2 bytes, then divide by 2³⁰."
hint: Multiply 2 (K and V) × layers × KV heads × head_dim × context × 2 bytes, then divide by 2³⁰ = 1,073,741,824 to get GiB.
```

```card
type: quiz
ref: no_mask
```

```card
type: quiz
ref: gqa_cache
```

```card
type: quiz
ref: why_not_q
```

## Build steps

### Cache one step

```challenge
id: cached_step
file: kv_cache.py
minutes: 40
start:
  code: |+
    import torch
    import torch.nn.functional as F

    # The tiny model from 2.1, split into pieces you can run one token at a time.
    # embed() adds token and position embeddings; head() turns the attention output into logits.
    text = "the cat sat on the mat. the dog sat on the log. a cat and a dog ran to the mat. "
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    encode = lambda s: torch.tensor([stoi[c] for c in s])
    decode = lambda ids: "".join(chars[i] for i in ids.tolist())


    class TinyLM:
        def __init__(self, seed=0):
            g = torch.Generator().manual_seed(seed)
            V, C, CTX = len(chars), 16, 64
            self.C = C
            counts = torch.ones(V, V) * 0.1
            for a, b in zip(text, text[1:]):
                counts[stoi[a], stoi[b]] += 1
            self.bigram = (counts / counts.sum(dim=1, keepdim=True)).log()
            self.wte = torch.randn(V, C, generator=g)
            self.wpe = torch.randn(CTX, C, generator=g) * 0.5
            self.Wq = torch.randn(C, C, generator=g) / C**0.5
            self.Wk = torch.randn(C, C, generator=g) / C**0.5
            self.Wv = torch.randn(C, C, generator=g) / C**0.5
            self.Wo = torch.randn(C, V, generator=g) * 0.15
            self.positions = 0  # how many positions embed() has processed, in total

        def embed(self, idx, pos):
            self.positions += torch.as_tensor(idx).numel()
            return self.wte[idx] + self.wpe[pos]

        def head(self, idx, h):
            return self.bigram[idx] + h @ self.Wo

        def __call__(self, idx):
            # 2.1's forward: every position at once, with a causal mask
            T = idx.shape[0]
            x = self.embed(idx, torch.arange(T))
            q, k, v = x @ self.Wq, x @ self.Wk, x @ self.Wv
            att = (q @ k.T) / self.C**0.5
            att = att.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf"))
            return self.head(idx, F.softmax(att, dim=-1) @ v)


    def generate(model, idx, max_new_tokens):
        # 2.1's greedy loop: re-runs the model on the whole sequence every step
        for _ in range(max_new_tokens):
            next_id = model(idx)[-1].argmax()
            idx = torch.cat([idx, next_id.view(1)])
        return idx


    model = TinyLM()
    prompt = encode("the ")

setup:
  - say: This is 2.1's tiny model, split into pieces. embed() adds the token and position embeddings, head() turns the attention output into logits, and 2.1's forward and greedy loop sit underneath.
    pause: 3.4
  - say: "Token t's key and value come only from tokens 0 to t, so they never change. Keep them in a cache, and each step only has to run the newest token."
    pause: 3.6
  - say: "Here's the stub. Your turn: grow the cache, then attend over it. The loop at the bottom checks every prompt position against the full forward."
    type_file: cached_step/stub.py
    pause: 1.2
tests: cached_step/tests.py
solution: cached_step/solution.py
alternatives:
  - cached_step/alternatives/unsqueeze.py
  - cached_step/alternatives/transpose.py
  - cached_step/alternatives/stack_rows.py
hints:
  logic: |
    embed the token at pos → q, k, v   (one row of C numbers each)
    make k and v one-row tables
      → add them under the cache's old rows
    score q against every cached key
      → ÷ √C → softmax → att   (one weight per cached token)
    mix the cached values using att → h
      → return the logits from h
  tools: |
    make a row into a one-row table
      t[None]              adds a new dimension of size 1
      t.unsqueeze(n)       same, at position n
    add it under the old rows
      torch.cat([a, b])    joins a and b along their first dimension
    score q against every cached key
      m @ v                dot product of each row of m with v
      plain arithmetic: /  and a square root of C
    scores to weights
      F.softmax(t, dim=-1) makes the last dimension add up to 1
    mix the cached values
      w @ m                weighted sum of the rows of m
  nudge: "Two jobs: remember this token's k and v, then let q look at everything remembered so far. What shape is the cache, and what shape is k?"
  concept: "The cache is a table with one row per token, and `k` is a single row of 16 numbers. Give it a row dimension and `torch.cat` it under the old rows. Attention is then the full forward's, for one query: dot q with every cached key (`cache[\"k\"] @ q`), divide by √C, softmax, and use the weights to mix the cached values."
  example: |
    Shrink the sizes so the numbers are readable: `C = 4`, so √C = 2. Two tokens are already cached, and token 2 arrives.

    - Old keys: `[[1, 0, 0, 0], [0, 1, 0, 0]]`. New `k = [1, 1, 0, 0]`. Giving `k` a row dimension makes it shape `[1, 4]`, and the cache grows from `[2, 4]` to `[3, 4]`.
    - Values grow the same way: old `[[1, 2, 3, 4], [5, 6, 7, 8]]`, new `v = [0, 0, 10, 10]`.
    - With `q = [2, 0, 0, 0]`, the dot products with the three keys are `[2, 0, 2]`. Divided by 2: `[1, 0, 1]`.
    - Softmax: `att ≈ [0.422, 0.155, 0.422]`. Three weights, one per cached token, adding up to 1.
    - Mixed values: `h ≈ [1.199, 1.777, 6.578, 7.155]`. It has `C = 4` numbers, like `q`.

    Check: the cache shape goes `[0, C]`, `[1, C]`, `[2, C]`... one row per step, and `att` always has as many entries as the cache has rows.
  skeleton: |
    cache["k"] = torch.___([cache["k"], k[____]])
    cache["v"] = torch.___([cache["v"], ____])
    att = F.softmax(cache["k"] @ q / model.C**____, dim=-1)
    h = ___ @ cache["___"]
  solution: |
    cache["k"] = torch.cat([cache["k"], k[None]])
    cache["v"] = torch.cat([cache["v"], v[None]])
    att = F.softmax(cache["k"] @ q / model.C**0.5, dim=-1)
    h = att @ cache["v"]
on_solve: Every position matches the full forward, and the cache grew by one row per token. Step 2 uses step() to generate.
mistakes:
  - { file: cached_step/mistakes/placeholders.py, fails: cache_grows, says: "`cache[\"k\"]` is still `...`" }
  - { file: cached_step/mistakes/h_placeholder.py, fails: cache_grows, says: "`h` is still `...`" }
  - { file: cached_step/mistakes/one_dim.py, fails: cache_grows, says: "a table and a single row of numbers" }
  - { file: cached_step/mistakes/overwrite.py, fails: cache_grows, says: "replaced the cache instead of adding to it" }
  - { file: cached_step/mistakes/keys_as_values.py, fails: cache_holds_kv, says: "holds the keys, not the values" }
  - { file: cached_step/mistakes/no_scale.py, fails: logits_match, says: "skip the division by √C" }
  - { file: cached_step/mistakes/scale_by_c.py, fails: logits_match, says: "divided by C = 16" }
  - { file: cached_step/mistakes/no_softmax.py, fails: logits_match, says: "the raw scaled scores" }
  - { file: cached_step/mistakes/no_main.py, fails: script_runs, says: "is missing" }
```

### Generate with the cache

```challenge
id: cached_generate
file: kv_cache.py
minutes: 30
start: { from: cached_step }
setup:
  - say: "Now the loop. First the prefill: push the prompt through step() one token at a time to fill the cache. After that, every new token is a single step."
    type_file: cached_generate/stub.py
    pause: 2.4
  - say: "The bottom runs both loops on fresh models and counts the positions each one processes. Fill in the two calls to step."
    pause: 1.2
tests: cached_generate/tests.py
solution: cached_generate/solution.py
alternatives:
  - cached_generate/alternatives/shape_pos.py
hints:
  logic: |
    cache starts empty
    prefill, for each prompt position pos:
      run the token at pos through step → logits
    repeat max_new_tokens times:
      next_id = position of the largest logit
      add next_id to the end of idx
      run ONLY next_id through step, at its index in idx
        → new logits
    return idx
  tools: |
    run one token through the model
      step(model, tok, pos, cache)
                           logits for token tok at position pos;
                           it also adds tok to the cache
    a prompt token
      t[i]                 item i of t
    where the newest token sits
      len(x)               how many items x has
  nudge: "Both blanks call step(). For each call: which token, at which position, with which cache?"
  concept: "In the prefill, token `idx[pos]` sits at position `pos`. In the loop, the id you just picked is already appended, so it's the last entry of `idx`, at index `len(idx) - 1`. Pass the same `cache` every time so the keys and values pile up."
  example: |
    Take a prompt `idx = [7, 3, 9]` and two new tokens.

    Prefill, 3 calls to `step`:

    - `step(model, 7, 0, cache)`: cache has 1 row.
    - `step(model, 3, 1, cache)`: cache has 2 rows.
    - `step(model, 9, 2, cache)`: cache has 3 rows. `logits` is the prediction after the `9`.

    Say it picks `4`. `idx` becomes `[7, 3, 9, 4]`, length 4. The new token sits at index 3, so `step(model, 4, 3, cache)`. Now the cache has 4 rows.

    Say it picks `6` next. `idx` is `[7, 3, 9, 4, 6]`, length 5, so `step(model, 6, 4, cache)`.

    Check: each token is run once, at its own index, so 3 prompt tokens plus 2 new ones is 5 positions processed. The no-cache loop re-reads the whole sequence each step: 3 + 4 = 7 positions for the same two tokens.
  skeleton: |
    logits = step(model, idx[___], ___, cache)
    logits = step(model, _______, len(idx) ___, cache)
  solution: |
    logits = step(model, idx[pos], pos, cache)
    logits = step(model, next_id, len(idx) - 1, cache)
on_solve: "Same text, 64 positions instead of 2,010. Each token now costs the same however long the story gets. The price is memory: the cache grows by one k and one v per token, in every layer. Step 3 measures that for a real model."
mistakes:
  - { file: cached_generate/mistakes/placeholders.py, fails: returns_tokens, says: "`logits` is still `...`" }
  - { file: cached_generate/mistakes/loop_placeholder.py, fails: returns_tokens, says: "`logits` is still `...`, the placeholder (line 91)" }
  - { file: cached_generate/mistakes/prefill_pos_zero.py, fails: matches_no_cache, says: "During the prefill the prompt tokens ran at positions" }
  - { file: cached_generate/mistakes/pos_zero.py, fails: matches_no_cache, says: "Every new token ran at position 0" }
  - { file: cached_generate/mistakes/pos_off_by_one.py, fails: matches_no_cache, says: "`len(idx)` counts the new token too" }
  - { file: cached_generate/mistakes/wrong_token.py, fails: matches_no_cache, says: "ran the token *before* the one you just picked" }
  - { file: cached_generate/mistakes/recompute.py, fails: one_position_per_token, says: "Something still runs the whole sequence" }
  - { file: cached_generate/mistakes/no_main.py, fails: script_runs, says: "are missing" }
```

### How big is the cache?

```challenge
id: kv_bytes
file: kv_cache.py
minutes: 20
start: { from: cached_generate }
setup:
  - say: On a big model the cache is the memory problem. Model configs list layers, query heads, KV heads and the hidden size, like Llama 3.1 8B's here.
    type_file: kv_bytes/stub.py
    pause: 2.4
  - say: "Your turn: write kv_cache_bytes. Llama 3.1 has 32 query heads but only 8 KV heads. Which count decides the cache size?"
    pause: 1.2
tests: kv_bytes/tests.py
solution: kv_bytes/solution.py
alternatives:
  - kv_bytes/alternatives/named.py
  - kv_bytes/alternatives/true_div.py
hints:
  logic: |
    head_dim = hidden_size ÷ number of query heads
    per position, in one layer, the cache holds:
      K and V   (2 tensors)
      each one head_dim-wide vector per KV head
        (KV heads, not query heads)
    multiply by layers, ctx positions and bytes_per_value
      → return the bytes
  tools: |
    read config values
      d[key]               the value stored under key
    head width
      plain arithmetic: //  whole-number division
    multiply the factors together
      plain arithmetic: *
  nudge: The cache holds a K and a V vector for every layer, every KV head and every position. How wide is each vector, and how many bytes does each number take?
  concept: "Each head is `hidden_size / num_attention_heads` wide, 128 for Llama 3.1 8B. With grouped-query attention, several query heads share one K/V head, so the cache holds `num_key_value_heads` heads, not `num_attention_heads`. Multiply 2 (K and V) × layers × KV heads × head width × positions × bytes per value."
  example: |
    Trace it on a small made-up config: 2 layers, 4 query heads, 2 KV heads, `hidden_size = 8`, `ctx = 3`, 2 bytes per value.

    - `head_dim`: 8 split across 4 heads is `2`.
    - K and V: `2`.
    - Times layers: `2 * 2 = 4`.
    - Times KV heads (2, not the 4 query heads): `4 * 2 = 8`.
    - Times `head_dim`: `8 * 2 = 16`.
    - Times `ctx`: `16 * 3 = 48` numbers.
    - Times bytes per value: `48 * 2 = 96` bytes.

    Check: if you used the 4 query heads you would get `192`, twice too much. The same slip on Llama 3.1 8B (32 query heads, 8 KV heads) is 4× too much: `4.00` GiB instead of `1.00`.
  skeleton: |
    head_dim = cfg["___________"] // cfg["___________________"]
    return ___ * cfg["num_hidden_layers"] * cfg["___________________"] * head_dim * ctx * _______________
  solution: |
    head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    return 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * head_dim * ctx * bytes_per_value
on_solve: "1.00 GiB per sequence at 8k. With all 32 heads cached it would be 4. Try ctx = 131072, Llama 3.1's full context: 16 GiB for one sequence, as much as the fp16 weights. That's why serving systems page the cache (5.2) and why people quantize it."
mistakes:
  - { file: kv_bytes/mistakes/placeholders.py, fails: returns_bytes, says: "still returns `...`" }
  - { file: kv_bytes/mistakes/return_placeholder.py, fails: returns_bytes, says: "still returns `...`" }
  - { file: kv_bytes/mistakes/gib.py, fails: returns_bytes, says: "that's GiB, already divided by 2**30" }
  - { file: kv_bytes/mistakes/bits.py, fails: returns_bytes, says: "that counts bits" }
  - { file: kv_bytes/mistakes/heads.py, fails: uses_kv_heads, says: "4.00 GiB, 4× too much" }
  - { file: kv_bytes/mistakes/hidden_size.py, fails: uses_kv_heads, says: "4.00 GiB, 4× too much" }
  - { file: kv_bytes/mistakes/head_dim_kv.py, fails: uses_kv_heads, says: "check the head width" }
  - { file: kv_bytes/mistakes/no_two.py, fails: every_factor, says: "two tensors per layer" }
  - { file: kv_bytes/mistakes/no_bytes.py, fails: every_factor, says: "That counts values, not bytes" }
  - { file: kv_bytes/mistakes/no_layers.py, fails: every_factor, says: "the size of one layer's cache" }
  - { file: kv_bytes/mistakes/no_main.py, fails: script_runs, says: "are missing" }
```

### Cache your 30M model

```replay
id: kv_cache_30m
intro: On your GPU, add a KV cache to your 30M model's attention and compare it with 2.1's loop on the same prompt. Watch a recording first.
log: replays/kv-cache-30m.jsonl
speed: 40
recorded:
  hardware: RTX 3090
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: ms per token with the cache, by position
  flag_above: 3.0
  x_labels: ["1–64", "449–512"]
  caption: Flat. Each token runs one position however long the story is. Without the cache the same run climbed from 3.1 to 7.0 ms per token.
summary:
  - { label: Speed, value: 216 → 480 tok/s }
  - { label: Logit match, value: max diff 1.9e-6 }
  - { label: KV cache at 512 tokens, value: 12.0 MiB }
run_yourself: |
  # on your machine or Colab, from the course repo
  python generate.py --ckpt runs/tinystories-30m/best.pt \
    --prompt "Once upon a time" --new-tokens 512 --temperature 0 --compare-cache
```

### Record your metrics

```metrics
id: record
intro: The logit match comes from your GPU run. The cache size comes from your kv_cache_bytes, for Llama 3.1 8B at 8k context.
```

## Hints

- On the GPU the cache has batch and head dimensions: `k` is `[B, n_kv_heads, T, head_dim]`, and new keys join along `dim=2`, the time axis.
- `torch.cat` copies the whole cache every step. Real implementations allocate `[B, H, max_len, D]` once and write each new token into its slot.
- With a cache the model's input is 1 token long, but positions keep counting. Pass the absolute position (often called `start_pos`) into the model, or every token gets position 0's embedding.
- If the cached logits differ by more than 1e-4 in fp32, compare layer by layer: the first layer that disagrees has the bug. In fp16, differences around 1e-3 are normal.

## Done when

- [ ] Cached logits match the full forward at every position {#cached_step auto=challenge:cached_step}
- [ ] Cached generation writes the same text with one position per token {#cached_generate auto=challenge:cached_generate}
- [ ] kv_cache_bytes gives 1 GiB for Llama 3.1 8B at 8k {#kv_bytes auto=challenge:kv_bytes}
- [ ] Logit match and KV cache size recorded {#metrics_recorded auto=metrics}
