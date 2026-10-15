---
gist: A freshly built network starts out confidently wrong, with its hidden units stuck where no gradient flows. Scaling the starting weights properly (Kaiming init) or normalising each layer (BatchNorm) fixes that, and writing the backward pass by hand shows what autograd does underneath.
minutes: 20
---

## What this lesson is for {#why}
> The makemore MLP trains much better when it starts from sensible weights. This lesson shows what goes wrong at step 0, two ways to fix it, and the backward pass behind it all.

0.3 ended with the bigram stuck near 2.45, because it only sees one letter. The fix is the model from Bengio et al.: look at three letters, through a small hidden layer. Karpathy calls it the makemore MLP, and it can get to about 2.10.

Built naively, though, it starts badly. Its first loss is about 27.9, far worse than the 3.30 of a model that knows nothing, and most of its hidden units are stuck. Training recovers, but it wastes thousands of steps and ends worse. This lesson covers:

1. **Why** a fresh network is confidently wrong, and how to see it.
2. **Kaiming init**: choose the starting weights' size so each layer keeps a healthy spread of values.
3. **BatchNorm**: normalise each layer's values during training, so the spread stays healthy.
4. **Manual backprop**: compute every gradient of a small MLP yourself and check it against autograd.

## What you need from before {#before}
> From 0.2: tanh, its slope 1 − tanh², and the chain rule. From 0.3: logits, softmax, the loss, and one-hot × matrix = row lookup.

| From | Idea |
|---|---|
| 0.0 | `sum(dim, keepdim=True)` and broadcasting; dim 0 runs down the rows |
| 0.1 | the uniform baseline, ln 27 ≈ 3.30: an honest "I don't know" |
| 0.2 | tanh squashes any number into −1…1; its slope is 1 − tanh², written with the *output* |
| 0.2 | chain rule: a value's gradient is the gradient from above times the local slope |
| 0.2 | a gradient adds up over every use of a value |
| 0.3 | logits → softmax → probabilities → average −ln p of the right answers |
| 0.3 | a one-hot vector times a matrix picks one row of that matrix |

## The makemore MLP {#mlp}
> Look up a learned 10-number vector for each of the last three letters, join them into 30 numbers, pass them through 200 tanh units, and score all 27 next letters.

Each input is the previous three characters, for example `"emm"` when predicting the `"a"` of "emma". For a batch of 32 such inputs, the shapes are:

| Step | Shape | What happens |
|---|---|---|
| input indices | `[32, 3]` | three character indices per example |
| embedding lookup in C `[27, 10]` | `[32, 3, 10]` | a learned 10-number vector per character |
| join (flatten) | `[32, 30]` | 3 × 10 = 30 numbers per example |
| `@ W1 + b1`, W1 `[30, 200]` | `[32, 200]` | pre-activations, `hpreact` |
| tanh | `[32, 200]` | the hidden layer, `h` |
| `@ W2 + b2`, W2 `[200, 27]` | `[32, 27]` | logits |
| softmax and loss | `[]` | one number |

The embedding table C is the same "one-hot × matrix picks a row" idea from 0.3, done directly by indexing. The parameters add up to 27·10 + 30·200 + 200 + 200·27 + 27 = **11,897** numbers.

A **layer** here means "multiply by a weight matrix, add a bias, maybe apply tanh". `fan_in` is how many inputs feed each unit of a layer: 30 for W1, 200 for W2.

## Problem 1: confidently wrong {#confident}
> Big random weights produce big logits, and softmax turns big logits into near-certainty. When that certainty is wrong, the loss is huge.

`torch.randn` draws numbers with a typical size of about 1. Multiply 200 hidden values by such weights and add them up, and logits easily reach ±10 or more. See what softmax does with that, using 27 classes where one wrong class has logit 10 and the rest have 0:

```
exp(10) ≈ 22,026
exp(0)  = 1   (26 of these)

p(wrong class) ≈ 22026 / 22052
               ≈ 0.9988
p(right class) ≈ 1 / 22052
               ≈ 0.000045
−ln 0.000045   ≈ 10.0
```

The model is 99.9% sure of a wrong answer, and the loss for that example is about 10. A fresh MLP produces logits even larger than this, and its starting loss is about 27.9.

An honest model at step 0 should spread its bets evenly and score about ln 27 ≈ 3.30. The fix: make the output layer's weights tiny (multiply W2 by 0.01) and its bias 0. The logits start near 0, softmax gives every class about 1/27, and the loss starts at about 3.32. Training then spends its steps learning instead of unlearning.

## Mean, variance and standard deviation {#variance}
> The mean is the average; the variance is the average squared distance from the mean; the standard deviation (std) is its square root, in the same units as the data.

The rest of the lesson talks about the "spread" of numbers, so here's how to measure it. Take the four numbers 2, 4, 6, 8:

```
mean      = (2+4+6+8) / 4 = 5
distances = −3, −1, 1, 3
squared   =  9,  1, 1, 9
variance  = (9+1+1+9) / 4 = 5
std       = √5 ≈ 2.236
```

The std says "values are typically about 2.2 away from the mean". In PyTorch: `t.mean()`, `t.var()`, `t.std()`, each taking a `dim` like `sum` does.

**n or n − 1?** The variance above divides by n = 4. PyTorch's `.var()` and `.std()` divide by **n − 1** by default (20 / 3 ≈ 6.67 here). That's Bessel's correction, a better estimate when your numbers are a sample from a larger population. BatchNorm in training uses the batch's own variance, dividing by n, so you pass `unbiased=False`. Getting this wrong makes every value come out a little small, by a factor like √(3/4) on a batch of 4, which one of the tests looks for.

## Problem 2: saturated tanh {#saturation}
> When tanh's input is large, its output sits at ±1, its slope is nearly 0, and almost no gradient flows back through that unit.

0.2 showed that tanh's slope is 1 − tanh². Look at what that does for inputs of different sizes:

| hpreact | h = tanh | slope 1 − h² |
|---|---|---|
| 0.5 | 0.462 | 0.79 |
| 3.0 | 0.995 | 0.0099 |
| −8.0 | −0.99999977 | 0.0000005 |

By the chain rule, every gradient passing back through a unit gets multiplied by that slope. A unit with input 3 passes on 1% of its gradient; one at −8 passes on almost nothing, so its incoming weights barely learn. Such a unit is **saturated**. The lesson counts a unit as saturated when |h| > 0.97.

With plain `randn` weights, W1's pre-activations have a spread of several units, and most hidden units start saturated. The lesson's illustrative run shows 79% at init.

A common slip in the backward pass, again: the slope uses tanh's *output* h, not its input hpreact. 1 − 3² = −8 is not a slope of tanh.

## Kaiming init: weights scaled by gain / √fan_in {#kaiming}
> A sum of fan_in random terms has a spread √fan_in times bigger than one term, so divide the weights by √fan_in, and multiply by a gain that makes up for what tanh squashes.

Each pre-activation is a sum of fan_in products, input × weight. When you add independent random numbers, their **variances** add (their stds don't). So with inputs and weights of std 1:

```
fan_in = 100 products, each var 1
variance of the sum = 100
std of the sum      = √100 = 10
```

A spread of 10 drives tanh deep into saturation. Scale the weights by 1/√fan_in and the sum's std comes back to about 1, whatever the layer's width.

tanh squashes, so a layer's outputs come out a bit smaller than its inputs. Over several layers that shrinking compounds. A **gain** makes up for it: 5/3 for tanh. That's the rule from He et al.'s paper (named Kaiming init after its first author), with PyTorch's gain for tanh:

```
std = gain / √fan_in
```

For W1, fan_in = 30: std = (5/3) / √30 ≈ 0.304. The first challenge stacks five tanh layers, 100 wide, and prints each layer's spread:

| Weights | Spread after 5 layers | Saturated |
|---|---|---|
| plain `randn` (std 1) | about 0.96 | about 83% |
| ÷ √fan_in, no gain | shrinks to about 0.32 | about 0% |
| × (5/3) / √fan_in | settles near 0.66 | about 6% |

Classic slips the tests catch: dividing by fan_in instead of its square root, using fan_out, forgetting the gain, using `torch.rand` (uniform from 0 to 1, so every weight is positive) instead of `torch.randn` (centred on 0), and not passing the seeded generator. The weight matrix is `[fan_in, fan_out]`, because the layer computes `h @ W` with h of shape `[batch, fan_in]`.

## BatchNorm {#batchnorm}
> For each feature, subtract its mean over the batch and divide by its std over the batch, then let the network rescale with a learned gamma and shift with a learned beta.

Kaiming init gets the spread right at step 0, but weights change as they train. **Batch normalisation** forces the spread every step. Take a batch of pre-activations, shape `[batch, features]`. For each feature (each column), over the examples in the batch (dim 0):

1. Subtract the column's mean.
2. Divide by √(variance + eps), its std with a tiny safety term.
3. Multiply by **gamma** and add **beta**, one of each per feature, learned like any weight.

One column, the numbers from the variance section:

```
column    = [2, 4, 6, 8]
mean 5, variance 5, std ≈ 2.236
normalised ≈ [−1.342, −0.447,
               0.447,  1.342]
```

Every column ends up with mean 0 and std 1. gamma and beta then let the network choose a different scale and offset if that helps; at the start gamma is 1 and beta 0, so nothing changes.

Details that matter, and that the tests check:

- **Down the batch, not across a row.** Normalising each example across its own features is a different layer, LayerNorm, which you'll meet in the transformer lessons.
- **eps** (like 0.00001) stops a division by zero when a feature is the same for every example.
- **Divide by the std, not the variance.** The variance is in squared units.
- **Biased variance**, dividing by n, to match PyTorch's `nn.BatchNorm1d` in training.

At **inference** you may feed a single name, and one example has no batch spread. So during training BatchNorm also keeps running averages of the means and variances, and `model.eval()` switches to them. And the bias b1 before a BatchNorm is pointless: subtracting the mean removes any constant you added, and beta takes its place.

## Backprop by hand: let the shapes guide you {#manual}
> A gradient always has the same shape as the thing it belongs to. For `out = x @ W + b`, that rule plus 0.2's product rule tells you which matrices to multiply, and the bias gradient sums over the batch.

The last challenge is a two-layer MLP small enough to check every number: 8 examples, 6 inputs, 10 hidden units, 5 classes. You write each gradient yourself, top to bottom, and compare it with PyTorch's.

**Every gradient has its tensor's shape.** `dW2` is shaped like W2, `dh` like h. This one rule settles most questions about transposes.

**Matrix multiply.** In 0.2, for out = x · w on single numbers, the local slope with respect to w was x, and with respect to x was w. A matmul is many of those products added up, so the gradients are matmuls of the *other* input with the incoming gradient, with one of them transposed (`.T` swaps rows and columns) so the shapes fit. Work it out from shapes: with x `[n, in]`, W `[in, out]` and the incoming gradient `[n, out]`, there's only one way to combine two of them into a `[in, out]` result, and one way to get `[n, in]`. If the shapes don't fit, PyTorch says "mat1 and mat2 shapes cannot be multiplied (5x4 and 5x6)", and the test tells you which shape you needed.

**Bias.** `b` has shape `[out]` but was added to every row of the batch, broadcast like in 0.0. A value used n times collects gradient from all n uses (0.2's "grads add up"), so the bias gradient **sums** the incoming gradient over the batch, dim 0. Summing over the other dim gives the wrong shape; averaging gives a value n times too small.

## Through softmax and tanh, and checking {#ce-tanh}
> The gradient of the loss with respect to the logits is (softmax − one-hot) / n. Through tanh it's elementwise: (1 − h²) times the incoming gradient.

**Cross-entropy.** `F.cross_entropy(logits, Y)` is 0.3's softmax plus average −ln p, in one call. Its gradient with respect to the logits has a neat form: in each row, the softmax probabilities, minus 1 at the right answer, all divided by n. With the softmax example from 0.3, where the right answer is class 1:

```
probs        [0.665, 0.245, 0.090]
minus 1 at 1 [0.665, −0.755, 0.090]
then / n
```

Read it as: push every wrong class down in proportion to how much probability it took, and push the right class up by how much it was missing. The tests check the three usual slips: forgetting `/ n` (every entry n times too big), forgetting to subtract the 1, and the sign flipped.

**Through tanh.** tanh acts on each number separately, so its backward does too: multiply elementwise (`*`, not `@`) by 1 − h², using the output h.

**Checking.** The script prints one line per gradient (wrapped here to fit):

```
W2 | exact: False |
     approximate: True |
     maxdiff: 1.5e-08
```

"exact: False" is fine. Your calculation adds things up in a different order from PyTorch's, and float32 keeps only about seven digits, so tiny differences around 0.00000001 are rounding, not bugs. "approximate: True" is what you want. The largest maxdiff is this lesson's systems metric; anything under 0.00001 passes, and a correct answer gives a few times 1e-8.

## The full run on your machine {#full-run}
> Train the MLP for 200,000 steps on all of names.txt with each fix. The original init ends near 2.17 on validation; every fix gets to about 2.10.

The lesson shows an **illustrative** replay of five 200k-step runs. It was written for the course, close to the numbers in Karpathy's makemore part 3, and it is not a recording. The setup: three characters of context, 10-number embeddings, 200 tanh units, batches of 32, learning rate 0.1 dropping to 0.01 after 100,000 steps, and an 80/10/10 split of the names into training, validation and test.

| Configuration | Step-0 loss | Validation loss |
|---|---|---|
| original (plain `randn`) | 27.88 | 2.168 |
| output layer × 0.01 | 3.32 | 2.130 |
| plus W1 × 0.2 (tanh unsaturated) | 3.31 | 2.103 |
| Kaiming (5/3) / √30 | 3.31 | 2.107 |
| BatchNorm | 3.31 | 2.105 |

Healthy signs in your own run: a step-0 loss near 3.3, few saturated units at init (the illustrative run goes from 79% to 4%), and a validation loss that ends near 2.10. Your exact numbers depend on the seed. Record your best validation loss as the quality metric; at most 2.15 passes. BatchNorm doesn't beat good init here, because a network this shallow is easy to initialise. In deep networks it matters much more.

## Words you'll see {#glossary}
> A short dictionary of the terms in this lesson's cards, challenges and metrics.

- **MLP (multi-layer perceptron)**: layers of "multiply by weights, add a bias, apply tanh", stacked.
- **Embedding**: a learned vector per character, looked up from a table (C).
- **Pre-activation (`hpreact`)**: a layer's values before tanh.
- **Hidden layer (`h`)**: the values after tanh, between the input and the output.
- **fan_in / fan_out**: how many inputs feed each unit of a layer / how many outputs it has.
- **Initialisation (init)**: how the weights are set before training.
- **Confidently wrong**: large logits that put almost all probability on a wrong answer; a step-0 loss far above ln 27.
- **Mean, variance, std**: the average; the average squared distance from it; the square root of the variance.
- **Bessel's correction**: dividing by n − 1 instead of n; `unbiased=False` turns it off.
- **Saturated**: a tanh unit stuck near ±1, with a slope near 0. Here, |h| > 0.97.
- **Kaiming init**: weights with std gain / √fan_in. The gain for tanh is 5/3.
- **Gain**: a factor that makes up for how much an activation function shrinks its input.
- **BatchNorm**: normalise each feature over the batch, then scale by gamma and shift by beta.
- **gamma, beta**: BatchNorm's learned scale and shift, one per feature.
- **eps**: a tiny number added to the variance so it's never 0.
- **Running mean / variance**: averages BatchNorm keeps during training and uses at inference.
- **LayerNorm**: normalises each example across its own features instead of across the batch.
- **Transpose (`.T`)**: swap a matrix's rows and columns.
- **Cross-entropy**: softmax plus average −ln p of the right answer, in one function.
- **dlogits, dW2, dh, …**: the gradient of the loss with respect to logits, W2, h, and so on.
- **maxdiff**: the largest gap between your gradient and autograd's; this lesson's systems metric.
- **Validation loss**: the loss on held-out names, this lesson's quality metric.
- **Illustrative replay**: example output written for the course to show what healthy looks like. Not a recording.
