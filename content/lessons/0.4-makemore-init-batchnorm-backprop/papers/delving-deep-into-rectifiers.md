---
paper: Delving Deep into Rectifiers
gist: Derive weight initialization from how variance flows through rectifier layers, so very deep ReLU networks train from scratch, and add a learnable leaky slope, PReLU; together they beat the reported human-level error on ImageNet.
---

## The problem {#problem}
> Very deep networks of rectified linear units were hard to train from scratch, because the usual initialization lets signals shrink or blow up layer by layer.

The common initialization of the time, Glorot and Bengio's, was derived for linear or symmetric activations. ReLU zeroes half its inputs, which halves the variance a layer passes on. Over a few layers that doesn't matter much; over dozens, the activations and gradients drift towards zero or towards huge values, and training stalls.

## PReLU {#prelu}
> Replace ReLU's flat negative half with a slope the network learns, one per channel, at almost no extra cost.

A parametric ReLU outputs x for positive inputs and a·x for negative ones, with a learned. A leaky ReLU fixes a at a small constant; PReLU trains it alongside the weights. The extra parameters are negligible, and the paper reports better accuracy than plain ReLU.

## The variance argument {#variance}
> Each output of a layer sums fan_in products of weights and inputs, so its variance is fan_in times the weight variance times the input's second moment.

Write one layer as y = W x. If the weights are independent with mean 0, the variance of y is the number of inputs, fan_in, times the variance of each weight times the mean square of each input. For that to stay the same from layer to layer, the weight variance has to cancel the factor of fan_in, and also make up for what the activation removes. For ReLU, which keeps half of a symmetric input, the factor works out to 2.

## The initialization {#init}
> Draw weights with std √(2 / fan_in): the extra factor of 2 makes up for ReLU zeroing half of its inputs.

The paper reaches the same form from the backward pass too, with fan_out in place of fan_in, and shows either version keeps the signal's scale stable through the network. For PReLU, the factor of 2 becomes 2 / (1 + a²). PyTorch names this Kaiming init after the first author and generalises the 2 into a gain per activation: √2 for ReLU, 5/3 for tanh. That's the gain / √fan_in you write in 0.4.

## What it achieves {#results}
> With this initialization a 30-layer network trains from scratch where Glorot initialization stalls, and the paper's ImageNet models reach 4.94% top-5 error, below the reported human level of 5.1%.

The deep-network experiment is the paper's direct evidence for the init. The ImageNet result combines PReLU, the init and wider, deeper models, and it was the first published result to beat the human-level figure on that benchmark.

## How the course uses it {#course}
> 0.4's `linear_weights` is this paper's rule with a tanh gain: scale normal weights by gain / √fan_in so activations keep their spread through every layer.

The makemore MLP uses tanh, not ReLU, so the gain changes but the reasoning doesn't: keep each layer's output spread near the input's, so tanh doesn't saturate and gradients survive. The five-layer stack in step 1 shows it directly: plain `randn` saturates almost every unit, and gain / √fan_in keeps the spread steady. In `torch.nn`, `nn.init.kaiming_normal_` does the same thing.
