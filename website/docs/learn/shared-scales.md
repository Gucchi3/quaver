---
title: "Shared scales"
description: "Give several tensors one scale, so that a device can add them as plain integers."
---

Two codes can only be added as integers if they have the same scale. Normally, each tensor has its own scale, so before an add, the device must first bring both inputs to one scale. Shared scales remove this step: several tensors get the same scale on purpose.

## The problem: adding two tensors

A residual add computes $y = a + b$. With codes $q_a$ and $q_b$ and scales $S_a$ and $S_b$:

$$
y = S_a\, q_a + S_b\, q_b
$$

If $S_a \ne S_b$, the codes cannot be added directly. The device must rescale at least one of them with a multiplier and a shift, and round, before it adds ([Residual add](residual-add.md)).

If $S_a = S_b = S$, the add is just:

$$
q_y = q_a + q_b \quad \text{(then clamp to the format)}
$$

One integer add, with no multiply, no shift, and no rounding.

## Scale sharing in a residual stream

A transformer block adds its branches to a residual stream: $x + \mathrm{attention}(x)$, then $x + \mathrm{MLP}(x)$. With scale sharing, one scale covers the whole stream inside the block:

- the input of the block,
- the output of the attention branch, and of the MLP branch,
- and both sums.

Each branch requantizes its last output to this shared scale. That requantization happens anyway, in the same step as the last conv of the branch, so it costs nothing extra. Then both adds are plain integer adds with saturation. The next block starts its own shared scale.

## How the shared scale is chosen

The shared scale must fit all its tensors at once, so it follows the widest of their ranges. During training, it keeps a moving average of the largest range seen in each pass. The tensors with a smaller range lose a little resolution. This is the price for adds that cost nothing.

## In Quaver

Make one `SharedScaler`, and give it to every quantizer and layer that should share the scale (with `output_scaler=` on the sample layers). The [CNN + ViT example](../examples/cnn-vit.md) shares one scale in the residual stream of each block.
