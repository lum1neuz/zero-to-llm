---
paper: Batch Normalization
gist: Normalize each layer's pre-activations over the mini-batch, then let the network rescale them with learned parameters, and deep networks train faster, with higher learning rates and less fuss about initialization.
---

## The problem {#problem}
> As earlier layers learn, the distribution of every later layer's inputs keeps shifting, which the authors call internal covariate shift.

Each layer is trained on inputs produced by the layers before it, and those keep changing as their weights update. The authors argue this forces small learning rates and careful initialization, and pushes saturating units like tanh and sigmoid into the flat regions where gradients vanish, the same saturation you see at init in 0.4.

## The transform {#transform}
> For each feature, subtract the mini-batch mean, divide by the mini-batch standard deviation (with a small epsilon), then scale by a learned gamma and shift by a learned beta.

This is the `batchnorm` you write in step 2. The statistics are computed per feature over the examples in the batch, using the batch's own variance. gamma and beta matter: with them, the layer can represent the identity, so normalizing never takes away something the network needed. A bias in the layer just before is redundant, because subtracting the mean cancels it and beta takes its place.

## Backpropagating through it {#backprop}
> The mean and variance depend on every example in the batch, so the gradient flows through them too, and each example's gradient depends on its batch-mates.

The paper derives the gradients through the normalization, including through the batch statistics, which makes BatchNorm part of the model rather than a preprocessing step. It also couples the examples in a batch: changing one example changes everyone's output. That coupling is one reason later architectures such as transformers use LayerNorm, which normalizes each example on its own.

## Training versus inference {#inference}
> At inference the batch statistics are replaced by population estimates collected during training, so the output for one input no longer depends on the batch.

After training, the layer uses fixed means and variances, estimated over the training data. In practice frameworks keep running averages during training, which is what `nn.BatchNorm1d` stores as `running_mean` and `running_var` and uses after `model.eval()`. Forgetting to switch modes is a classic bug.

## Convolutional layers {#conv}
> For convolutions, normalize each channel over the batch and over every spatial position, so the same feature is normalized the same way everywhere in the image.

Each channel gets one gamma and one beta, shared across positions, which keeps the convolutional property that a feature means the same thing wherever it appears.

## What it achieves {#results}
> On ImageNet, a batch-normalized Inception network reaches the original's accuracy in many times fewer training steps, and an ensemble improved on the best published result.

Batch normalization allowed much higher learning rates, made training less sensitive to the initial weights, made saturating nonlinearities usable in deep networks, and acted as a regularizer, reducing the need for dropout. Later work questioned whether reducing internal covariate shift is really why it helps, pointing instead at a smoother optimization landscape. The practical benefits aren't in dispute.

## How the course uses it {#course}
> 0.4 writes the training-mode transform by hand and checks it against nn.BatchNorm1d; the transformer lessons switch to LayerNorm.

In the makemore MLP, BatchNorm between the first linear layer and tanh gets the validation loss to about the same place as careful Kaiming init. In a network that shallow it's a safety net more than a necessity. From phase 1 on, every block uses LayerNorm, with the same normalize-then-gamma-and-beta shape but statistics over each token's features instead of over the batch.
