---
gist: A brand-new network starts out sure of wrong answers, with many of its parts stuck. Two fixes, choosing sensible starting weights and BatchNorm, get it off to a good start. Then you work out its gradients yourself to see what autograd does.
minutes: 25
---

## Start here {#why}
> A bigger model that looks at three letters can beat 0.3's score, but only if it starts out sensibly. This lesson shows what goes wrong at the start and how to fix it.

In 0.3 the letter-pair model got stuck near 2.45, because it only ever looks at one letter. A model that looks at the **last three letters** can do better, down to about 2.10. It comes from a 2003 paper by Bengio and colleagues, and Karpathy builds it in his makemore videos. We'll call it the **makemore MLP**.

But if you build it the obvious way, it starts badly. Imagine a new student who answers every question with total confidence, and is usually wrong. Before they can learn anything, they first have to unlearn that confidence. Training wastes thousands of steps doing exactly that.

This lesson covers:

1. **What goes wrong** when a new network starts.
2. **Fix 1, Kaiming init**: choose the size of the starting weights so numbers stay a sensible size as they pass through the network.
3. **Fix 2, BatchNorm**: rescale the numbers inside the network on every training step.
4. **Backprop by hand**: work out every gradient of a small network yourself, and check it against PyTorch.

## What you need from before {#before}
> You need tanh and its slope, and the chain rule from 0.2, plus scores, softmax and the loss from 0.3.

| Idea | Reminder | From |
|---|---|---|
| Rows and `dim` | `dim=0` runs down the rows of a grid; `keepdim=True` keeps the shape for dividing | 0.0 |
| Baseline | a model that knows nothing scores about 3.30 (ln 27) | 0.1 |
| tanh | squashes any number into −1…1. Its slope is 1 − (its output)². | 0.2 |
| Chain rule | along a chain of steps, the slopes multiply | 0.2 |
| Grads add up | a number used several times collects grad from every use | 0.2 |
| Scores to chances | logits, then softmax, then the average surprise at the right answers | 0.3 |
| One-hot | a one-hot list times a table picks one row of the table | 0.3 |

## The makemore MLP {#mlp}
> Each of the last three letters gets a list of 10 learned numbers. The 30 numbers go through 200 tanh units, and then become a score for each of the 27 possible next letters.

**MLP** stands for "multi-layer perceptron". It just means a network made of **layers**. Each layer multiplies by a table of weights, adds a bias, and usually applies tanh.

Here's what happens to one example, the three letters "emm", when guessing the "a" in "emma":

1. **Look up each letter.** Each of the 27 symbols has its own list of 10 learned numbers, stored in a table called **C**. This is the one-hot "pick a row" idea from 0.3. Such a list is called an **embedding**.
2. **Join them.** Three lists of 10 make one list of 30.
3. **Layer 1.** Multiply by a weight table W1 and add a bias b1. That gives 200 numbers, called **pre-activations** (`hpreact`).
4. **tanh.** Squash each of the 200 numbers. These form the **hidden layer** (`h`).
5. **Layer 2.** Multiply by W2 and add b2. That gives 27 scores (logits), one per possible next letter.

Training uses 32 examples at once, called a **batch**. The shapes for a batch:

| Step | Shape | Meaning |
|---|---|---|
| letter numbers | `[32, 3]` | 3 letters for each of 32 examples |
| look up in C `[27, 10]` | `[32, 3, 10]` | 10 numbers per letter |
| join | `[32, 30]` | 30 numbers per example |
| times W1 `[30, 200]`, plus b1 | `[32, 200]` | pre-activations |
| tanh | `[32, 200]` | hidden layer |
| times W2 `[200, 27]`, plus b2 | `[32, 27]` | scores |

**fan_in** means how many numbers feed into each unit of a layer: 30 for layer 1, and 200 for layer 2.

## Problem 1: too sure of itself {#confident}
> Big random starting weights make big scores, and softmax turns big scores into near-certainty. When that certainty is wrong, the loss is huge.

`torch.randn` picks random numbers that are usually between −2 and 2. Multiply 200 of them by the hidden values and add them all up, and the scores can easily reach 10 or more.

See what softmax does with a big score. Say one wrong letter gets a score of 10, and the other 26 letters, including the right one, get 0:

| Step | Wrong letter | Each other letter |
|---|---|---|
| score | 10 | 0 |
| exp | about 22,026 | 1 |
| chance (divide by the total, about 22,052) | 0.9988 | 0.000045 |

The model is 99.9% sure of a wrong letter. The surprise at the right letter is −ln 0.000045, about 10. A brand-new MLP produces scores even bigger than this, and it starts with a loss of about **27.9**.

An honest beginner would spread its chances evenly and score about **3.30**. The fix is simple: make the last layer's starting weights tiny, by multiplying W2 by 0.01, and start b2 at 0. Then all the scores start near 0, softmax gives every letter about the same chance, and the loss starts at about **3.32**. Now training can spend its steps learning.

## Measuring spread {#variance}
> The mean is the average. The variance is the average squared distance from the mean. The standard deviation (std) is the square root of the variance, and says how far numbers typically are from the mean.

The rest of the lesson talks about how spread out some numbers are. Here's how to measure it, with four numbers: 2, 4, 6, 8.

| Step | Working | Result |
|---|---|---|
| mean (average) | (2 + 4 + 6 + 8) / 4 | 5 |
| distances from the mean | 2−5, 4−5, 6−5, 8−5 | −3, −1, 1, 3 |
| squared distances | 9, 1, 1, 9 | |
| variance: their average | (9 + 1 + 1 + 9) / 4 | 5 |
| standard deviation (std) | the square root of 5 | about 2.236 |

The std says "these numbers are typically about 2.2 away from their average". Squaring the distances stops the negatives cancelling out the positives; the square root then brings the answer back to the same units as the numbers.

In PyTorch: `t.mean()`, `t.var()` and `t.std()`. Each takes a `dim`, like `sum` does.

**Dividing by 4 or by 3?** Above, we divided by 4, the number of values. PyTorch's `.var()` and `.std()` divide by one less, 3, unless you tell them otherwise (that gives 20 / 3 ≈ 6.67). That's called **Bessel's correction**. It's a better guess when your numbers are a small sample from a much larger group. BatchNorm, below, uses the plain version, dividing by the number of values. You get it with `unbiased=False`. If you forget, every result comes out slightly too small, and a test points this out.

## Problem 2: stuck units {#saturation}
> When tanh gets a big number, its output sits at −1 or 1, and its slope is nearly 0. Almost no gradient gets back through it, so it stops learning.

0.2 showed that tanh's slope is 1 minus its output squared. Look at what that means for different inputs:

| Input (`hpreact`) | Output (`h`) | Slope, 1 − h² |
|---|---|---|
| 0.5 | 0.462 | 0.79 |
| 3.0 | 0.995 | 0.0099 |
| −8.0 | −0.99999977 | 0.0000005 |

By the chain rule, every gradient passing back through a unit gets multiplied by that slope. A unit whose input is 3 lets only 1% of its gradient through. A unit at −8 lets almost nothing through, so the weights feeding it hardly change. A unit stuck like this is called **saturated**. The lesson counts a unit as saturated when its output is beyond ±0.97.

With plain random weights, layer 1's numbers are far too spread out, so most hidden units start saturated. The lesson's illustrative run shows 79% of them stuck at the start.

A mistake to watch for: the slope uses tanh's *output*, h. Using the input instead gives nonsense: 1 − 3² = −8 isn't a slope of tanh at all.

## Fix 1: Kaiming init {#kaiming}
> Adding up many random numbers makes a bigger spread. Shrink the starting weights by the square root of fan_in, and multiply by a small "gain" for tanh, and the spread stays steady from layer to layer.

**Init**, short for initialisation, means how you set the weights before training starts.

Each pre-activation is a sum of fan_in products, input times weight. When you add up random numbers, their spread grows. The rule is that their **variances** add up. With 100 products, each with variance 1:

| | Value |
|---|---|
| variance of one product | 1 |
| variance of the sum of 100 | 100 |
| std of the sum | the square root of 100 = 10 |

A spread of 10 pushes tanh straight into saturation. So we shrink the weights by the square root of fan_in. Then the sum's spread comes back to about 1, however wide the layer is.

tanh also squashes its input, so each layer's output comes out a bit smaller than its input. Over several layers that shrinking adds up. A **gain** makes up for it. For tanh the gain is 5/3, about 1.67. Put together, the starting weights get this spread:

```
std = gain / square root of fan_in
```

This rule comes from a 2015 paper by Kaiming He and colleagues, so it's called **Kaiming init**. (PyTorch supplies the gain of 5/3 for tanh.)

For layer 1, fan_in is 30: 1.67 / 5.48 ≈ **0.304**.

The first challenge stacks five tanh layers, each 100 wide, and prints how spread out each layer's numbers are:

| Starting weights | Spread after 5 layers | Saturated |
|---|---|---|
| plain `randn` (std 1) | about 0.96 | about 83% |
| shrunk by √fan_in, no gain | falls to about 0.32 | about 0% |
| gain / √fan_in | stays near 0.66 | about 6% |

The middle row shows why the gain matters: without it, the numbers shrink a little at every layer.

Slips the tests catch: dividing by fan_in instead of its square root, using fan_out (the number of outputs), leaving out the gain, using `torch.rand` (which only gives positive numbers between 0 and 1) instead of `torch.randn` (centred on 0), and not using the seeded generator. The weight table has shape `[fan_in, fan_out]`, because the layer works out `h @ W` with h shaped `[batch, fan_in]`.

## Fix 2: BatchNorm {#batchnorm}
> For each feature, subtract its average over the batch and divide by its spread over the batch. Then let the network rescale with two learned numbers, gamma and beta.

Kaiming init gets the spread right at the start, but the weights change as they learn. **Batch normalisation**, or **BatchNorm**, fixes the spread on every training step.

Take a batch of pre-activations, shaped `[batch, features]`. A **feature** is one column: one of the 200 units. For each column, using all the examples in the batch:

1. Subtract the column's mean.
2. Divide by its std. (In fact by the square root of variance + **eps**, a tiny number like 0.00001.)
3. Multiply by **gamma** and add **beta**. These are learned, one of each per column.

Here's one column, using the numbers from "Measuring spread":

| Value | Minus the mean (5) | Divided by the std (2.236) |
|---|---|---|
| 2 | −3 | −1.342 |
| 4 | −1 | −0.447 |
| 6 | 1 | 0.447 |
| 8 | 3 | 1.342 |

Every column ends up with an average of 0 and a spread of 1. gamma and beta then let the network choose a different spread and centre if that helps. They start at 1 and 0, which changes nothing.

Details the tests check:

- **Down each column, not across each row.** Normalising each example across its own 200 numbers is a different tool, called LayerNorm. You'll meet it in the transformer lessons.
- **eps** stops a division by zero when a column has the same value in every example.
- **Divide by the std, not the variance.** The variance is in squared units.
- **Divide by n**, the plain variance (`unbiased=False`), to match PyTorch's `nn.BatchNorm1d`.

**Using the model later.** When you make up names, you may give the model one example at a time, and one example has no spread. So during training, BatchNorm also keeps running averages of each column's mean and variance. Calling `model.eval()` switches it to use those.

One last detail: the bias b1 just before a BatchNorm does nothing useful. Subtracting the mean removes any fixed number you added, and beta takes its place.

## Backprop by hand: shapes show the way {#manual}
> A gradient always has the same shape as the thing it belongs to. That rule, plus the product rule from 0.2, tells you what to multiply.

The last challenge uses a small network: 8 examples, 6 inputs, 10 hidden units and 5 possible answers. You work out each gradient yourself, from the loss back to the first layer, and compare with PyTorch.

**Rule 1: a gradient has its tensor's shape.** The gradient for W2, written `dW2`, has the same shape as W2. `dh` has the same shape as h. This one rule answers most questions about which way round things go.

**Rule 2: grid multiplication.** In 0.2, for c = a × b with single numbers, the slope for a was b, and the slope for b was a. A grid multiplication is lots of those products added up. So its gradients are grid multiplications too: the *other* input times the gradient coming back, with one of them **transposed** so the shapes fit. Transposing, written `.T`, swaps rows and columns:

```
m = [[1, 2, 3],
     [4, 5, 6]]   shape [2, 3]

m.T = [[1, 4],
       [2, 5],
       [3, 6]]    shape [3, 2]
```

Use the shapes like a puzzle. Say the input x is `[n, in]`, the weights W are `[in, out]`, and the gradient coming back is `[n, out]`. There's only one way to combine two of them into an `[in, out]` result, and only one way to get `[n, in]`. If the shapes don't fit, PyTorch says something like "mat1 and mat2 shapes cannot be multiplied (5x4 and 5x6)", and the test tells you which shape you needed.

**Rule 3: the bias.** b has shape `[out]`, but it was added to every row of the batch, by broadcasting. A number used n times collects gradient from all n uses (0.2's "grads add up"). So the bias gradient **adds up** the gradient coming back, down the batch. Adding along the other direction gives the wrong shape. Averaging gives an answer n times too small.

## Through softmax and tanh {#ce-tanh}
> The gradient of the loss for the scores is: the chances, minus 1 at the right answer, all divided by the number of examples. Through tanh, multiply each number by 1 − h².

**Through the loss.** `F.cross_entropy(logits, Y)` does 0.3's softmax and average surprise in one go. Its gradient for the scores has a neat form, in words:

1. Take the chances from softmax.
2. In each row, subtract 1 at the right answer.
3. Divide everything by the number of examples, n.

With 0.3's softmax example, where the right answer is the second letter:

| | Letter 1 | Letter 2 (right) | Letter 3 |
|---|---|---|---|
| chances | 0.665 | 0.245 | 0.090 |
| minus 1 at the right one | 0.665 | −0.755 | 0.090 |
| then divide by n | | | |

Read it like this: push each wrong letter down by how much chance it took, and push the right letter up by how much chance it was missing. The tests check the three usual slips: forgetting to divide by n, forgetting to subtract the 1, and getting the sign the wrong way round.

**Through tanh.** tanh works on each number separately, so its gradient does too. Multiply each number of the gradient coming back by 1 − h² for that unit, using the output h. That's an ordinary `*`, not the grid multiplication `@`.

**Checking.** The script prints one line per gradient. Here's one, wrapped to fit:

```
W2 | exact: False |
     approximate: True |
     maxdiff: 1.5e-08
```

"exact: False" is fine. Your calculation adds numbers up in a different order from PyTorch's, and float32 only keeps about 7 digits. So tiny differences, around 0.00000001, are rounding, not mistakes. "approximate: True" is what you want. Your biggest maxdiff is this lesson's second score. Anything under 0.00001 passes, and a correct answer gives a few times 0.00000001.

## The long run on your computer {#full-run}
> Train the MLP for 200,000 steps on all the names, once per fix. The original start ends near 2.17 on held-back names; each fix gets to about 2.10.

The lesson shows an **illustrative** replay of five long training runs. It was written for the course, close to the numbers in Karpathy's makemore videos. It's not a recording of a real run.

The setup: three letters of context, 10 numbers per letter, 200 tanh units, batches of 32, and a learning rate of 0.1 that drops to 0.01 halfway through. The names are split 80% for training, 10% for validation and 10% for a final test.

| Setup | Loss at step 0 | Validation loss |
|---|---|---|
| original (plain `randn`) | 27.88 | 2.168 |
| last layer × 0.01 | 3.32 | 2.130 |
| plus W1 × 0.2, so tanh isn't stuck | 3.31 | 2.103 |
| Kaiming init (5/3) / √30 | 3.31 | 2.107 |
| BatchNorm | 3.31 | 2.105 |

Healthy signs in your own run: a loss near 3.3 at step 0, few saturated units at the start (the illustrative run goes from 79% to 4%), and a validation loss that ends near 2.10. Your exact numbers depend on the seed. Record your best validation loss as this lesson's quality score; 2.15 or less passes.

BatchNorm doesn't beat a good init here, because this network has only one hidden layer. In much deeper networks it matters a lot more.

## Words you'll see {#glossary}
> A short list of the new words in this lesson's cards, challenges and scores.

- **MLP (multi-layer perceptron)**: a network made of layers: multiply by weights, add a bias, apply tanh.
- **Layer**: one "multiply by weights, add a bias" step, usually followed by tanh.
- **Embedding**: a learned list of numbers for each letter, looked up from a table (C).
- **Batch**: a group of examples processed together, here 32.
- **Pre-activation (`hpreact`)**: a layer's numbers before tanh.
- **Hidden layer (`h`)**: the numbers after tanh, between the input and the output.
- **fan_in, fan_out**: how many numbers feed into each unit of a layer, and how many come out.
- **Init (initialisation)**: how the weights are set before training.
- **Mean, variance, std**: the average; the average squared distance from it; the square root of the variance.
- **Bessel's correction**: dividing by one less than the count. `unbiased=False` turns it off.
- **Saturated**: a tanh unit stuck near −1 or 1, with a slope near 0. Here, beyond ±0.97.
- **Kaiming init**: starting weights with spread gain / √fan_in. The gain for tanh is 5/3.
- **Gain**: a factor that makes up for how much tanh squashes its input.
- **BatchNorm**: rescale each feature to average 0 and spread 1 over the batch, then apply gamma and beta.
- **Feature**: one column of a layer's numbers, one unit.
- **gamma, beta**: BatchNorm's learned stretch and shift, one of each per feature.
- **eps**: a tiny number added to the variance so you never divide by zero.
- **Running mean, running variance**: averages BatchNorm keeps during training, for use afterwards.
- **LayerNorm**: like BatchNorm, but across each example's own numbers instead of down the batch.
- **Transpose (`.T`)**: swap a grid's rows and columns.
- **Cross-entropy**: softmax plus the average surprise at the right answer, in one function.
- **dlogits, dW2, dh, …**: the gradient of the loss for the logits, for W2, for h, and so on.
- **maxdiff**: the biggest gap between your gradient and PyTorch's. This lesson's second score.
- **Validation loss**: the loss on held-back names. This lesson's quality score.
- **Illustrative**: example output written for the course to show what healthy looks like. Not a recording.
