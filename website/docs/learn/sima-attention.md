---
title: "SimA attention"
description: "An attention with no softmax: L1-normalized queries and keys, which a device can compute with integer math only."
---

The attention of a transformer uses softmax, which needs the exponential function and a float division for every score. These are hard to compute with integers. SimA (Simple Softmax-free Attention) replaces softmax with a much simpler normalization.

## The idea

In normal attention:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

SimA drops softmax, and normalizes $Q$ and $K$ with the L1 norm (the sum of the absolute values) instead. Quaver normalizes each token (row) of $Q$ and of $K$ by its own L1 norm:

$$
\hat{Q} = \frac{Q}{\lVert Q \rVert_1}, \qquad \hat{K} = \frac{K}{\lVert K \rVert_1}, \qquad \mathrm{SimA}(Q, K, V) = \hat{Q}\, \hat{K}^\top V
$$

The normalization keeps the scores in a fixed range, which is what softmax did, so training stays stable. There is no exponential, and the division happens only once for each token.

## Integer math: normalize and quantize in one step

The L1 norm depends on the input, so the device must compute it at run time. NEQ (normalization-equivalent quantization) merges the normalization into the quantization. Dividing a token by its L1 norm is the same as dividing its scale by the L1 norm. So the device:

1. Adds up the absolute values of the codes of a token, to get its L1 norm (an integer).
2. Finds the multiplier and the shift for the ratio "scale ÷ L1 norm" with an integer division.
3. Requantizes the token with them.

All of this is integer math. One scale can be found for each token, or one for the whole tensor of a head, which is cheaper on a device.

## The rest of the attention

- $\hat{Q} \hat{K}^\top$ is an integer matrix product of codes. Its result is requantized to the format of the attention map, with $1/\sqrt{d}$ folded into the fixed part of the multiplier.
- The attention map times $V$ is another integer matrix product, followed by a requantization.
- A 1×1 conv projects the result, as in normal attention.

## Why it suits devices

- No exponential, no lookup table for it, and no float division.
- The whole block runs on the same integer multiply-add hardware as the convs.
- Without softmax, the product can also be computed as $\hat{Q} (\hat{K}^\top V)$. For many tokens, this order is cheaper, since $\hat{K}^\top V$ is a small $d \times d$ matrix.

## In Quaver

`QuantSimaAttention` runs SimA with NEQ, with integer math in both modes. The [CNN + ViT example](../examples/cnn-vit.md) uses it in two transformer blocks.
