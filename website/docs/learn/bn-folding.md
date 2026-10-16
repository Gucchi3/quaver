---
title: "BatchNorm folding"
description: "Merge a BatchNorm into the conv before it, so the device runs one layer instead of two."
---

After training, the parameters of a BatchNorm (BN) are fixed numbers. So a BN right after a conv or linear layer can be merged into that layer's weights and bias. This is called folding.

## The math

A linear layer and a BN after it compute:

$$
y = W x + b, \qquad z = \frac{\gamma}{\sigma}\,(y - \mu) + \beta, \qquad \sigma = \sqrt{v + \epsilon}
$$

Here $\mu$ and $v$ are the running mean and variance of the BN, and $\gamma$ and $\beta$ are its learned scale and shift. Put $y$ into $z$:

$$
z = \underbrace{\frac{\gamma}{\sigma}\,W}_{W_{fold}}\, x \;+\; \underbrace{\frac{\gamma}{\sigma}\,(b - \mu) + \beta}_{b_{fold}}
$$

So $z = W_{fold}\,x + b_{fold}$: one layer with new weights and a new bias. For a conv, $\gamma / \sigma$ scales each output channel of the weight.

## Why fold before quantizing

- The device runs one layer instead of two, and needs no BN at all.
- The weights that the device uses are the folded ones. If you quantize the weights before folding, folding changes them again after quantization, and the codes no longer fit the format.

So the usual order is: train in float, fold every BN, load the folded weights into the quantized model, and then run QAT.

## In Quaver

`fold_batch_norms(model)` returns a copy of a float model with every BN folded into the conv right before it. The examples load these folded weights into their quantized models. The [CNN + ViT example](../examples/cnn-vit.md) even uses BN in place of LayerNorm, so that every normalization folds away.
