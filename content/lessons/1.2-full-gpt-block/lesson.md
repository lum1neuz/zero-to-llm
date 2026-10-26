---
id: "1.2"
title: Full GPT block
summary: Split attention into heads, wrap it with LayerNorms, an MLP and residuals into a GPT-2 block, and count its parameters by hand.
phase: "1"
hours: 6
type: build
status: ready
protected: false
cut: null
prerequisites:
  - "1.1"
papers:
  - title: Attention Is All You Need
    authors: Vaswani et al.
    year: 2017
    url: https://arxiv.org/abs/1706.03762
  - title: Language Models are Unsupervised Multitask Learners
    authors: Radford et al.
    year: 2019
quality_metric:
  id: val_loss
  label: Shakespeare validation loss
  unit: nats
  note: 6 blocks, 384 wide, 5,000 steps at a constant learning rate. The small CPU config (4 blocks, 128 wide) lands near 1.9.
  inputs:
    - { id: val_loss, label: Val loss at step 5000, hint: The last eval line of your run }
  compute: value
  reference: 1.52
  pass: { max: 1.95 }
systems_metric:
  id: params_m
  label: Parameter count
  unit: M
  note: "Check it by hand: embeddings + 6 × block_params(384) + the final LayerNorm."
  inputs:
    - { id: params_m, label: Parameters (millions), hint: The training script prints it at start-up }
  compute: value
  reference: 10.77
  pass: { max: 11.0 }
quiz:
  - id: why_residual
    q: Why does each block compute x = x + f(x) instead of x = f(x)?
    answer: 1
    options:
      - text: The addition doubles what the block can represent
        why: The addition adds no parameters and no capacity. Its job is about training, not expressiveness.
      - text: Gradients reach the early layers directly through the additions, so deep stacks still train
        why: The derivative of x + f(x) with respect to x is 1 + f′(x). That 1 is a clean path from the loss back to the first block, however many blocks sit in between.
      - text: It keeps the activations from growing
        why: It does the opposite. Each block adds to x, so the residual stream tends to grow. Keeping each sublayer's input in range is LayerNorm's job.
      - text: Without it, attention could see the future
        why: Causality comes from the mask inside attention. The residual path carries each token's own vector forward, never another position's.
  - id: why_heads
    q: n_embd is 384. Why use 6 heads of 64 instead of one head of 384?
    answer: 0
    options:
      - text: Each head can look at different tokens for different reasons, for about the same cost
        why: One head makes one weighted average per token. Six heads make six, each from its own q and k, so one can track the subject while another tracks the last comma. c_attn is d × 3d either way.
      - text: Six heads have six times the parameters, so the model is stronger
        why: The parameter count is the same. c_attn maps 384 → 1152 whether you cut the 1152 into one head or six.
      - text: Smaller heads are causal and one big head isn't
        why: The causal mask works the same at any head size.
      - text: The 1/√d scale only works for small heads
        why: The scale works at any size. That's the point of dividing by √d.
---

## Goal

Turn 1.1's single head into a full **GPT-2 block**. Split attention into several **heads** that run side by side, put a **LayerNorm** in front of attention and of a new **MLP**, and add each one's output back onto x through a **residual** connection. Then predict the block's **parameter count** without building it, and see where the weights of a GPT live.

## Learn

```card
type: concept
id: block_layout
eyebrow: The block
title: Two sublayers, each wrapped the same way.
visual:
  kind: bars
  max: 2.2
  bars:
    - { label: attention, value: 1.05, text: 1.05M }
    - { label: MLP, value: 2.1, text: 2.10M }
    - { label: LayerNorms, value: 0.002, text: 0.002M }
body: |
  `x = x + attn(ln_1(x))`, then `x = x + mlp(ln_2(x))`. Attention lets tokens exchange information; the MLP then works on each token by itself, widening it to 4 × n_embd and back. Each sublayer reads a **normalised copy** of x and **adds** its result onto x, the residual stream. The bars show where one block's parameters live at n_embd = 512: the MLP holds two thirds.
```

```card
type: predict
id: predict_heads_shape
eyebrow: Predict the output
title: B = 2, T = 10, C = 32, 4 heads. What prints?
code: |
  hs = C // 4
  q = q.view(B, T, 4, hs).transpose(1, 2)
  print(q.shape)
options:
  - text: torch.Size([2, 10, 4, 8])
    why: That's after the view but before the transpose. The transpose moves the head axis in front of T.
  - text: torch.Size([2, 4, 10, 8])
    correct: true
    why: "view splits each token's 32 features into 4 heads of 8: [2, 10, 4, 8]. transpose(1, 2) swaps T and the heads: [2, 4, 10, 8]. Now each head is its own batch entry of 10 tokens."
  - text: torch.Size([2, 4, 8, 10])
    why: That swaps the last two axes, which is what you do to k inside attention. Here dims 1 and 2 swap.
  - text: torch.Size([8, 4, 10, 2])
    why: That reverses everything. transpose swaps exactly two axes and leaves the others where they are.
hint: Do it in two steps. First the view, then swap the axes at positions 1 and 2.
```

```card
type: fill_blank
id: block_forward
eyebrow: Fill the blanks
title: Complete the pre-norm block.
code: |
  def forward(self, x):
      x = x + self.attn(self.[[0]](x))
      x = [[1]] + self.mlp(self.ln_2(x))
      return x
tokens: [ln_1, ln_2, x, self.ln_2(x), self.attn(x)]
answer: [ln_1, x]
tips:
  ln_2: "`ln_2` belongs to the MLP. Attention gets its own LayerNorm, `ln_1`, with its own learned gain and bias."
  self.ln_2(x): Adding onto a normalised x would throw away the residual stream's scale every block. Only the sublayer's *input* is normalised.
  self.attn(x): "The first line already added attention's output into x. Adding it again would count it twice."
why_ok: "Each sublayer reads its own normalised copy and adds its result onto the untouched residual stream x."
why_wrong: "The residual stream x itself is never replaced, only added to. And each sublayer has its own LayerNorm."
hint: The second line has the same shape as the first, with the MLP and its own LayerNorm.
```

```card
type: calc
id: gpt2_block_params
eyebrow: Do the math
title: How many parameters are in one GPT-2 small block?
given:
  - { label: n_embd d, value: "768" }
  - { label: Per block, value: "12·d² + 13·d" }
unit: M
answer: { min: 7.0, max: 7.2 }
placeholder: "0.0"
why_ok: 12 × 768² = 7,077,888 weights plus 13 × 768 = 9,984 biases and LayerNorm values, 7.09M. GPT-2 small stacks 12 of them, 85M of its 124M parameters.
why_wrong: Square 768 first (589,824), multiply by 12, then add the small 13·d term.
hint: 768² is 589,824. The 13·d term barely moves the answer.
```

```card
type: reorder
id: multi_head_order
eyebrow: Put it in order
title: Arrange multi-head attention's forward pass.
lines:
  - q, k, v = self.c_attn(x).split(C, dim=2)
  - q = q.view(B, T, nh, hs).transpose(1, 2)  # k and v the same way
  - y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
  - y = y.transpose(1, 2).reshape(B, T, C)
  - return self.c_proj(y)
shuffle: [3, 1, 4, 0, 2]
why_ok: "Make q, k and v in one matmul, give each head its own axis, attend per head, put the heads back side by side, then mix them with c_proj."
why_wrong: "Follow the shapes: [B, T, 3C] → [B, nh, T, hs] → attention → [B, T, C] → c_proj."
hint: The heads have to be split before attention and merged after it.
```

```card
type: quiz
ref: why_residual
```

```card
type: quiz
ref: why_heads
```

## Build steps

### Split attention into heads

```challenge
id: multi_head
file: block.py
minutes: 45
start: { code: "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n" }
setup:
  - say: In 1.1 you wrote one head. A GPT runs several heads side by side, each on its own slice of the features, and makes q, k and v for all of them in one matmul.
    pause: 3.4
  - say: "Here's the layer. c_attn makes q, k and v at once and split cuts them apart. The fused attention call does 1.1's work for every head. Your turn: give each head its own axis, then put the heads back together."
    type_file: multi_head/stub.py
    pause: 1.2
tests: multi_head/tests.py
solution: multi_head/solution.py
alternatives:
  - multi_head/alternatives/reshape_minus_one.py
  - multi_head/alternatives/permute.py
  - multi_head/alternatives/contiguous_view.py
  - multi_head/alternatives/flatten.py
hints:
  start:
    try: |
      q, k, v = attn.c_attn(x).split(32, dim=2)
      print(tuple(q.shape))
      print(tuple(q[0, 3].view(4, 8).shape))
    see: |
      (2, 8, 32)
      (4, 8)
    next: "That's `q` as it arrives, then one token's 32 numbers cut into 4 heads of 8. Do that cut for every token at once, then move the heads axis in front of the tokens. The fused attention call wants `(2, 4, 8, 8)`: batch, heads, tokens, features."
  logic: |
    for each of q, k, v:
      cut its C features into n_head chunks of hs
        → move the head axis in front of T
        → [B, n_head, T, hs]
    attention runs once per head   (given)
    put the heads back:
      move the head axis behind T again
        → merge n_head × hs into C
        → y is [B, T, C]
  tools: |
    cut the features into heads
      t.view(a, b, c, d)    same data, new shape
    move the head axis in front of T
      t.transpose(i, j)     swaps axes i and j of t
    put the heads back, merge into C
      t.transpose(i, j)     swaps axes i and j of t
      t.reshape(a, b, c)    same data, new shape
  nudge: "Each of q, k and v is [B, T, C], and C is really n_head slices of hs features. Make the slices an axis of their own, then move that axis in front of T so attention runs once per head."
  concept: "`view(B, T, n_head, hs)` splits each token's C features into n_head chunks of hs, keeping every token's features together. `transpose(1, 2)` then swaps T and the head axis: [B, n_head, T, hs], so attention treats each head like a batch entry. On the way back, swap them again and merge the last two axes into C."
  example: |
    Trace it on a tiny case: `B = 1`, `T = 2`, `C = 4`, `n_head = 2`, so `hs = 2`. Let `q` be `[[0,1,2,3], [4,5,6,7]]`, one row per token:

    - Cut into heads: token 0 is `[0,1]` and `[2,3]`, token 1 is `[4,5]` and `[6,7]`. Shape `[1, 2, 2, 2]`.
    - Swap `T` and the head axis: head 0 sees `[[0,1], [4,5]]` (both tokens, first 2 features), head 1 sees `[[2,3], [6,7]]`. Shape `[1, 2, 2, 2]` again, but now it reads `[B, n_head, T, hs]`.
    - Coming back, swap again, then merge: `[[0,1,2,3], [4,5,6,7]]`, shape `[1, 2, 4]`.

    Check: if you skip the swap-back, the merge glues the wrong things together. Token 0 would come out as `[0,1,4,5]`. Print the shape after each step.
  skeleton: |
    q = q.____(B, T, self.n_head, __).transpose(_, _)
    k = k.____(B, T, self.n_head, __).transpose(_, _)
    v = v.____(B, T, self.n_head, __).transpose(_, _)
    y = y.transpose(_, _).reshape(_, _, ___)
  solution: |
    q = q.view(B, T, self.n_head, hs).transpose(1, 2)
    k = k.view(B, T, self.n_head, hs).transpose(1, 2)
    v = v.view(B, T, self.n_head, hs).transpose(1, 2)
    y = y.transpose(1, 2).reshape(B, T, C)
on_solve: "Each head now sees only its own 8 features, and all 4 run in one batched call. Step 2 is unlocked: wrap this attention and an MLP into a block."
mistakes:
  - { file: multi_head/mistakes/placeholders.py, fails: output_shape, says: "`q` is still `...`" }
  - { file: multi_head/mistakes/y_placeholder.py, fails: output_shape, says: "`y` is still `...`" }
  - { file: multi_head/mistakes/k_not_transposed.py, fails: output_shape, says: "q, k and v in different layouts" }
  - { file: multi_head/mistakes/split_no_transpose.py, fails: no_future, says: "`view(B, n_head, T, hs)` cuts the [T, C] block in reading order" }
  - { file: multi_head/mistakes/merge_no_transpose.py, fails: no_future, says: "Swap the head axis and T back first" }
  - { file: multi_head/mistakes/hs_before_heads.py, fails: matches_reference, says: "n_head comes before hs" }
```

### Assemble the block

```challenge
id: block
file: block.py
minutes: 40
start: { from: multi_head }
setup:
  - say: Attention lets tokens exchange information. The MLP then works on each token by itself, widening it to 4 × n_embd, through a GELU, and back.
    type_file: block/stub.py
    pause: 2.8
  - say: "Your turn: two LayerNorms, and a forward pass where each sublayer reads a normalised x and adds its result back onto x."
    pause: 1.2
tests: block/tests.py
solution: block/solution.py
alternatives:
  - block/alternatives/named_steps.py
  - block/alternatives/tuple_shape.py
hints:
  start:
    try: |
      ln = nn.LayerNorm(32)
      print(tuple(ln(x).shape))
      print(tuple(attn(ln(x)).shape))
    see: |
      (2, 8, 32)
      (2, 8, 32)
    next: "Both keep the shape `(2, 8, 32)`, the same as `x`. That's why a sublayer's result can be added straight back onto `x`. Write the attention line first: normalise a copy, run `attn`, add it onto `x`. The MLP line is the same pattern with its own LayerNorm."
  logic: |
    when the block is built:
      ln_1 = a LayerNorm for the attention side
      ln_2 = a LayerNorm for the MLP side
    when x comes in:
      normalise a copy of x with ln_1 → attn
        → add the result onto x   (x itself stays raw)
      normalise a copy of the new x with ln_2 → mlp
        → add the result onto x
      → return x   (same shape as the input)
  tools: |
    build the two LayerNorms
      nn.LayerNorm(n)     normalises each token's n features
    run a sublayer on a normalised copy
      layer(x)            applies a layer or sublayer to x
    add the result onto x
      plain arithmetic: +
  nudge: "Two lines, the same pattern twice. Each sublayer gets a normalised copy of x, and whatever it returns is *added* to x. Which LayerNorm goes with which sublayer?"
  concept: "`nn.LayerNorm(n_embd)` normalises each token's n_embd features and learns a gain and a bias. The residual stream x is never normalised itself: `x + self.attn(self.ln_1(x))` leaves a direct path from the block's input to its output, which is what lets deep stacks train."
  example: |
    Trace one token with 4 features, `x = [1, 2, 3, 4]`. Pretend the sublayers return fixed numbers:

    - `ln_1(x)` ≈ `[−1.342, −0.447, 0.447, 1.342]`: mean 0, spread 1. This copy goes into `attn`, and `x` itself stays untouched.
    - Say `attn` returns `[0.5, 0, 0, −0.5]`. Added onto the original: `x = [1.5, 2, 3, 3.5]`.
    - `ln_2(x)` ≈ `[−1.265, −0.632, 0.632, 1.265]`. This copy goes into `mlp`.
    - Say `mlp` returns `[0.1, 0.1, 0.1, 0.1]`. Added onto `x`: `[1.6, 2.1, 3.1, 3.6]`.

    Check: the output is never a normalised vector. It's the raw `x` plus two small corrections. For `Block(32, 4)` the printed shape is the input's shape, `(2, 8, 32)`.
  skeleton: |
    self.ln_1 = nn.________(______)
    self.ln_2 = nn.________(______)
    x = x + self.____(self.____(x))
    x = _ + self.mlp(self.____(x))
  solution: |
    self.ln_1 = nn.LayerNorm(n_embd)
    self.ln_2 = nn.LayerNorm(n_embd)
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
on_solve: "That's a GPT-2 block. Stack 6 of them between the embeddings and a final LayerNorm and you have the model you train next. Step 3: predict its parameter count without building it."
mistakes:
  - { file: block/mistakes/placeholders.py, fails: layer_norms, says: "`self.ln_1` is still `...`" }
  - { file: block/mistakes/batchnorm.py, fails: layer_norms, says: "BatchNorm averages each feature across the *batch*" }
  - { file: block/mistakes/shared_ln.py, fails: layer_norms, says: "the *same* LayerNorm" }
  - { file: block/mistakes/forward_placeholders.py, fails: keeps_shape, says: "`x` is still a placeholder" }
  - { file: block/mistakes/no_residual.py, fails: residual, says: "nothing carries x past the sublayers" }
  - { file: block/mistakes/post_norm.py, fails: residual, says: "normalising the residual stream itself" }
  - { file: block/mistakes/no_ln_attn.py, fails: pre_norm, says: "Attention reads x raw" }
  - { file: block/mistakes/norm_output.py, fails: pre_norm, says: "You normalise attention's *output*" }
  - { file: block/mistakes/ln1_twice.py, fails: pre_norm, says: "Your MLP reads `ln_1`'s output" }
  - { file: block/mistakes/parallel.py, fails: pre_norm, says: "the two sublayers run side by side" }
  - { file: block/mistakes/swapped_ln.py, fails: pre_norm, says: "You swapped the LayerNorms" }
```

### Count its parameters

```challenge
id: block_params
file: block.py
minutes: 20
start: { from: block }
setup:
  - say: Your block printed its parameter count. Could you have predicted it without building anything?
    type_file: block_params/stub.py
    pause: 2.4
  - say: "The loop builds a real block at two widths and compares. Your turn: count attention, the MLP and the LayerNorms in terms of d."
    pause: 1.2
tests: block_params/tests.py
solution: block_params/solution.py
alternatives:
  - block_params/alternatives/collected.py
  - block_params/alternatives/powers.py
hints:
  start:
    try: |
      c = block.attn.c_attn
      print(tuple(c.weight.shape), tuple(c.bias.shape))
      print(c.weight.numel() + c.bias.numel())
    see: |
      (96, 32) (96,)
      3168
    next: "That's one layer counted by hand: a weight grid plus one bias per output. With `d = 32` this layer maps 32 numbers to 96. Write that count with `d` in place of 32, then do the same for the other layers in `attn`, `mlp` and `norms`."
  logic: |
    for each linear layer, count weights + biases:
      weights = in × out
      biases  = out
    attn = c_attn (d → 3d) + c_proj (d → d)
    mlp  = c_fc (d → 4d) + c_proj (4d → d)
    for each of the 2 LayerNorms:
      gain d + bias d
    norms = both LayerNorms added up
    attn + mlp + norms
      → return the total   (the script compares it to the built block)
  tools: |
    weights + biases of each linear layer
      plain arithmetic: * +   (in × out, then + out)
    the two LayerNorms
      plain arithmetic: * +   (gain d, bias d, twice)
    add attn + mlp + norms
      plain arithmetic: +
  nudge: "Every nn.Linear(in, out) holds in × out weights plus out biases. Every LayerNorm(d) holds a gain and a bias, d each. List the six layers and add them up."
  concept: "Attention: c_attn is Linear(d, 3d), c_proj is Linear(d, d). MLP: c_fc is Linear(d, 4d), c_proj is Linear(4d, d). Then two LayerNorms of width d. Write each layer as its weights plus its biases."
  example: |
    Trace it with a tiny width, `d = 4`:

    - `c_attn`: 4 × 12 weights + 12 biases = 60. `c_proj`: 4 × 4 weights + 4 biases = 20. So `attn = 80`.
    - `c_fc`: 4 × 16 weights + 16 biases = 80. `c_proj`: 16 × 4 weights + 4 biases = 68. So `mlp = 148`.
    - Two LayerNorms of 4 + 4 each: `norms = 16`.
    - Total: 80 + 148 + 16 = `244`.

    Check: that's 12·4² + 13·4 = 192 + 52 = 244. At `d = 32` the same count gives `12,704`, the number the script's `built` column should match.
  skeleton: |
    attn = _ * d * d + _ * d + d * d + ___  # c_attn and c_proj
    mlp = d * _ * d + _ * d + _ * d * d + ___  # c_fc and c_proj
    norms = _ * (___ + ___)  # ln_1 and ln_2
  solution: |
    attn = 3 * d * d + 3 * d + d * d + d  # c_attn and c_proj
    mlp = d * 4 * d + 4 * d + 4 * d * d + d  # c_fc and c_proj
    norms = 2 * (d + d)  # ln_1 and ln_2
on_solve: "12·d² + 13·d. At d = 512 that's 3,152,384 per block, so the 6 blocks of 1.4's 30M model hold 18.9M of its 30.2M parameters, and most of the rest is the token embedding. The 13·d part is under 0.3%: the matrices are the model."
mistakes:
  - { file: block_params/mistakes/placeholders.py, fails: returns_number, says: "`attn` is still `...`" }
  - { file: block_params/mistakes/mlp_narrow.py, fails: weights, says: "The MLP widens each token to 4·d features" }
  - { file: block_params/mistakes/c_attn_single.py, fails: weights, says: "c_attn makes q, k *and* v in one matmul" }
  - { file: block_params/mistakes/no_attn_proj.py, fails: weights, says: "Attention has two Linear layers" }
  - { file: block_params/mistakes/no_norms.py, fails: biases_and_norms, says: "not the LayerNorms" }
  - { file: block_params/mistakes/gain_only.py, fails: biases_and_norms, says: "a gain *and* a bias" }
  - { file: block_params/mistakes/bias_per_input.py, fails: biases_and_norms, says: "one bias per *output*" }
  - { file: block_params/mistakes/no_biases.py, fails: biases_and_norms, says: "The Linear biases are missing" }
  - { file: block_params/mistakes/plus_one.py, fails: biases_and_norms, says: "constant term of 1" }
```

### Train it on Shakespeare

```replay
id: shakespeare_char
intro: "Stack 6 of your blocks, 384 wide, and train a character-level GPT on Tiny Shakespeare at a constant learning rate. On a GPU it takes about 6 minutes; on a laptop CPU, use the small config and expect about 20. Here's what a healthy GPU run looks like."
log: replays/shakespeare-char.jsonl
speed: 40
recorded:
  hardware: RTX 3090
  illustrative: true
  note: Sample data written for the course, shaped like nanoGPT's Shakespeare-char run. Not a real recording. Replace it with a log of your own run.
chart:
  title: Validation loss every 500 steps
  flag_above: 2.4
  x_labels: [step 0, step 5000]
  caption: Only step 0 is worse than 1.1's single head (2.40). By step 500 the full model is already well past it. Watch val flatten while train keeps falling; 1.3 is about getting the most out of those last steps.
summary:
  - { label: Val loss, value: 2.40 → 1.52 }
  - { label: Parameters, value: 10.77M }
  - { label: Train time, value: 6 min }
run_yourself: |
  # on your machine or Colab, from the course repo
  python train_gpt.py --config shakespeare_char \
    --max-iters 5000 --lr 1e-3 --schedule constant
  # laptop CPU: add --n-layer 4 --n-embd 128 --block-size 64
```

### Record your metrics

```metrics
id: record
intro: Your val loss and parameter count from your own run. They're compared with the reference run and the pass limit.
```

## Hints

- After `transpose`, real PyTorch can't `view` a tensor whose memory is no longer in order. Use `reshape`, or `.contiguous().view(...)`. The browser's mini-torch lets `view` through, so your GPU is where it would fail.
- Split C as `(n_head, hs)`, never `(hs, n_head)`: each head owns one contiguous slice of features, which is how GPT-2's weights are laid out.
- Pre-norm (LayerNorm before each sublayer) is what GPT-2 and every later GPT use. The 2017 paper normalised after the residual add, which is harder to train deep.
- The model's parameter count is `wte + wpe + n_layer × block_params(d) + 2·d` for the final LayerNorm. With weight tying, the output layer reuses `wte` and adds nothing.

## Done when

- [ ] Multi-head attention keeps each token's features together {#multi_head auto=challenge:multi_head}
- [ ] Your block passes x straight through when its sublayers are off {#block auto=challenge:block}
- [ ] block_params matches a built block {#block_params auto=challenge:block_params}
- [ ] Val loss and parameter count recorded {#metrics_recorded auto=metrics}
