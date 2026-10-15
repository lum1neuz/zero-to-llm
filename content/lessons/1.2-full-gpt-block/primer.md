---
gist: A GPT block is attention split into several heads, followed by a small per-token network, with each part reading a normalised copy of the input and adding its result back on top.
minutes: 18
---

## What this lesson is for {#why}
> One attention head lets tokens look at each other. A GPT block adds the pieces that make a stack of those trainable and useful.

In 1.1 you built one head. A real GPT doesn't use one big head. It uses several small ones side by side, then gives every token a short private "think" step, and wraps both in two tricks that let you stack dozens of blocks without training falling apart.

By the end of this lesson you have the exact block GPT-2 uses. Stack 6 of them between the embeddings and the output layer and you have the 10.77M-parameter Shakespeare model in this lesson's recording. Make them 512 wide and you have the 30M model you train in 1.4 and quantize in phase 3. The last step counts a block's parameters by hand, which is how you'll size that model before spending GPU time on it.

## What you need from before {#before}
> Your 1.1 head, the [B, T, C] shape convention, and how nn.Linear stores its weights.

- **x has shape `[B, T, C]`**: B sequences, T tokens each, C features per token.
- **A head** makes q, k and v from x, scores every query against every earlier key, and returns a weighted mix of values.
- **`nn.Linear(in, out)`** holds a weight of shape `[out, in]` and, by default, a bias of `out` numbers.
- **Softmax** turns a row of scores into weights that add up to 1, and the causal mask hides the future.

In this lesson the attention itself comes from `F.scaled_dot_product_attention(q, k, v, is_causal=True)`, PyTorch's ready-made version of what you wrote by hand in 1.1. You've earned the shortcut.

## Several heads instead of one {#heads}
> Cut each token's C features into n_head slices and run a separate attention on each slice. Same cost, more kinds of looking.

One head computes one weighted average per token. That means one notion of "relevant": maybe "the subject of this verb". But a token often needs several things at once: the subject, the last punctuation mark, the word it rhymes with. Several heads let each one learn its own pattern.

The trick is that the heads are narrower. With C = 32 and 4 heads, each head works on 8 features. Four heads of 8 do about the same amount of work as one head of 32, so you get variety without paying for it.

GPT-2 also makes q, k and v for all heads in **one** matmul. A layer called `c_attn` maps C features to 3·C, and `split` cuts the result into three `[B, T, C]` tensors: q, k and v. Each of those still holds every head's slice side by side. The next section is about pulling the slices apart.

## Giving each head its own axis {#split}
> Reshape so each token's features are cut into heads, then move the head axis in front of T. Done in the wrong order, tokens get mixed up.

Take one sequence of 2 tokens, C = 4 features, and 2 heads of 2 features. Token 0's features are `a0 a1 a2 a3` and token 1's are `b0 b1 b2 b3`. Head 0 should get the first two features of every token, head 1 the last two:

```
want head 0: [a0 a1]   (token 0)
             [b0 b1]   (token 1)
want head 1: [a2 a3]   (token 0)
             [b2 b3]   (token 1)
```

Two moves get there:

| Step | Shape | What it does |
|---|---|---|
| start | `[B, T, C]` | one row of C per token |
| view as `(B, T, n_head, hs)` | `[B, T, nh, hs]` | each token's row cut into heads |
| swap axes 1 and 2 | `[B, nh, T, hs]` | heads in front, tokens inside |

After the swap, attention sees n_head separate little sequences, one per head, and treats the head axis like part of the batch.

**The classic mistake** is to view straight to `(B, n_head, T, hs)`. `view` only reinterprets numbers in memory order, and memory holds `a0 a1 a2 a3 b0 b1 b2 b3`. Read as `[2 heads, 2 positions, 2 features]`, that gives:

```
head 0: [a0 a1]   "position 0"
        [a2 a3]   "position 1"
head 1: [b0 b1]   "position 0"
        [b2 b3]   "position 1"
```

Head 1's "position 0" is really token 1. When token 0's output is assembled, it includes a head that read token 1, a token from the future. The shapes are all correct and nothing crashes. The test that changes the last tokens and watches the earlier outputs catches it.

## Putting the heads back {#merge}
> Undo the two moves in reverse: swap the head axis and T back, then merge the last two axes into C.

The attention output has shape `[B, nh, T, hs]`. Swap axes 1 and 2 to get `[B, T, nh, hs]`, where each token's heads sit together again, then merge the last two axes into one of size C. A final `c_proj` layer, C to C, mixes the heads' results.

Skipping the swap and reshaping straight to `[B, T, C]` has the same problem as before: output position 0 collects head 0's positions 0, 1 and 2 instead of every head's position 0.

One difference between the browser and your GPU: in real PyTorch, `view` fails on a tensor that has just been transposed, because its numbers are no longer stored in order. Use `reshape`, or `.contiguous()` before `view`. The browser's mini-torch lets `view` through, so your GPU is where you'd find out.

## The MLP {#mlp}
> After attention mixes information between tokens, a small two-layer network processes each token on its own: widen to 4·C, apply GELU, shrink back to C.

Attention is the only place tokens talk to each other. The **MLP** ("multi-layer perceptron") is where each token does something with what it heard. It's the same network applied to every position separately.

| Layer | Shape change |
|---|---|
| `c_fc` | C → 4·C |
| GELU | same size, bends values |
| `c_proj` | 4·C → C |

**GELU** is a smooth cousin of ReLU. ReLU cuts every negative number to 0. GELU lets small negatives through, shrunk:

| x | ReLU(x) | GELU(x) |
|---|---|---|
| −2 | 0 | −0.05 |
| −1 | 0 | −0.16 |
| 0 | 0 | 0 |
| 1 | 1 | 0.84 |
| 2 | 2 | 1.95 |

Without a bend like this, two linear layers in a row would collapse into one, and the MLP would add nothing. The MLP class is given to you in step 2.

## LayerNorm {#layernorm}
> LayerNorm rescales each token's features to mean 0 and variance 1, then applies a learned gain and bias per feature.

Take one token with features `[2, 4, 6, 8]`:

```
mean      = 5
diffs     = [-3, -1, 1, 3]
variance  = (9+1+1+9) / 4 = 5
std       = √5 ≈ 2.236
normed    = [-1.34, -0.45, 0.45, 1.34]
```

Then each feature is multiplied by its own learned **gain** (starts at 1) and shifted by its own **bias** (starts at 0). So `nn.LayerNorm(C)` holds 2·C parameters. PyTorch also adds a tiny `eps` inside the square root so a constant row doesn't divide by zero.

Compare BatchNorm from 0.4. **BatchNorm** averages each feature *across the batch*, so one sequence's output depends on what else is in the batch. **LayerNorm** averages *across one token's own features*, so every token is normalised on its own. Language models use LayerNorm, and a step 2 test tells you so if you reach for BatchNorm.

## Residual connections {#residual}
> Each sublayer's output is added to its input: x = x + f(x). The sum, called the residual stream, carries information straight through the whole stack.

Think of the residual stream as a shared document passed down a line of editors. Each sublayer reads it and *adds* a note; nobody rewrites the page. If a sublayer has nothing useful to say, it adds roughly zero and the document passes through unchanged.

That's also why deep stacks train. The derivative of `x + f(x)` with respect to x is `1 + f′(x)`. The 1 is a clean path for the gradient from the loss back to the first block, however many blocks sit in between.

The step 2 tests use exactly this idea. They zero out the last layer of attention and of the MLP, so both sublayers output 0, and check that your block returns x unchanged. If it returns all zeros, nothing carries x past the sublayers. If it returns x normalised, you're normalising the stream itself.

## Pre-norm: where the LayerNorm goes {#prenorm}
> Normalise the sublayer's input, not the residual stream. Each sublayer gets its own LayerNorm.

GPT-2's block, in words:

1. Normalise x with the first LayerNorm, run attention on that, and add the result to x.
2. Normalise the new x with the second LayerNorm, run the MLP on that, and add the result to x.

This is **pre-norm**. The 2017 Transformer paper put the LayerNorm *after* the addition (post-norm), normalising the whole stream each time. That works for a few layers but gets harder to train deep, and every GPT since GPT-2 uses pre-norm.

| Variant | What's wrong |
|---|---|
| post-norm | the stream itself gets normalised |
| no norm before attention | attention reads raw x |
| norm on attention's output | the norm is on the wrong side |
| MLP reads `ln_1` | sublayers share one gain |
| MLP reads x from before attention | the sublayers run side by side |
| one LayerNorm object used twice | no separate gain and bias |

The "side by side" layout is a real design (GPT-J uses it on purpose), but it isn't GPT-2's.

## Counting parameters {#params}
> A Linear(in, out) layer holds in·out weights plus out biases. A LayerNorm(d) holds 2·d. Add up a block's six layers and you get its size without building it.

For example, `c_attn` at C = 32 maps 32 → 96: that's 32 × 96 = 3,072 weights plus 96 biases, 3,168 parameters. Biases belong to the *outputs*, one per output feature.

A block has six layers to count: `c_attn` (d → 3d), attention's `c_proj` (d → d), `c_fc` (d → 4d), the MLP's `c_proj` (4d → d), and two LayerNorms of width d. Step 3 asks you to add those up as a formula in d. The script then builds real blocks at d = 32 and d = 512 and compares. At d = 32 a block has 12,704 parameters, so you can check your formula by hand before running it.

The finished formula has a d² part, from the matrices, and a much smaller d part, from biases and LayerNorms. At d = 512 the small part is under 0.3% of the total: the matrices *are* the model.

## Training it on Shakespeare {#run}
> Six blocks, 384 wide with 6 heads, trained for 5,000 steps at a constant learning rate, reach a validation loss of about 1.52.

The recording shows a character-level GPT on Tiny Shakespeare: 6 blocks, n_embd 384, 6 heads, a 256-character context, 10.77M parameters, about 6 minutes on an RTX 3090. It's **illustrative**: sample data written for the course, shaped like nanoGPT's Shakespeare run, not a real recording.

Healthy looks like this. The loss starts near 4.29 and passes 1.1's single head (2.40) within the first 500 steps. Validation loss then flattens around 1.52 while training loss keeps falling to about 1.19. That growing gap means the model has started to memorise its training text. 1.3 is about getting more out of those last steps.

On a laptop CPU, use the small config (4 blocks, 128 wide, 64-character context). It takes about 20 minutes and lands near 1.9, which still passes. Your metrics are your val loss (pass under 1.95) and the parameter count the script prints (reference 10.77M).

## Words you'll see {#glossary}
> The terms in this lesson's cards, challenges and metrics.

- **Block**: one GPT layer: attention and an MLP, each with a LayerNorm and a residual connection.
- **Multi-head attention**: several attention heads run side by side on separate slices of the features.
- **n_head, hs (head size)**: the number of heads, and the features per head, C / n_head.
- **c_attn**: the layer that makes q, k and v for every head at once, C → 3·C.
- **c_proj**: the output layer of attention (and, separately, of the MLP).
- **view / reshape**: give the same numbers a new shape. `view` needs them stored in order on a GPU.
- **transpose**: swap two dimensions.
- **MLP**: the per-token network, C → 4·C → C.
- **c_fc**: the MLP's widening layer.
- **GELU**: a smooth activation that lets small negative values through.
- **LayerNorm**: normalises each token's features, then applies a learned gain and bias.
- **BatchNorm**: normalises each feature across the batch (0.4). Not used in GPTs.
- **Residual connection / residual stream**: x = x + f(x), and the running sum it creates.
- **Pre-norm / post-norm**: LayerNorm before each sublayer (GPT-2) or after the addition (2017).
- **ln_1, ln_2**: the block's two LayerNorms, in front of attention and of the MLP.
- **Parameter count**: the number of learned numbers. Linear(in, out) has in·out + out, LayerNorm(d) has 2·d.
