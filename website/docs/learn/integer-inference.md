---
title: "Integer inference"
description: "How a quantized layer runs on a device with integers only: products, INT32 sums, the bias, and requantization."
---

On a device with integer hardware, a quantized model can run with integers only, from the input pixels to the output. This page follows one layer.

## The idea

Each real value is a code times a scale: $r = S\,(q - Z)$. Put this into a linear layer $y = \sum_i w_i\, x_i + b$, with symmetric weights:

$$
y = \sum_i S_w q_{w,i} \cdot S_x (q_{x,i} - Z_x) + b = S_w S_x \Big( \underbrace{\sum_i q_{w,i}\,(q_{x,i} - Z_x)}_{\text{integers}} + \frac{b}{S_w S_x} \Big)
$$

The sum is pure integer math. The scales stay outside, as one factor $S_w S_x$. So the device adds up integer products, and deals with the scales only once, at the end.

## The steps of one layer

1. **Multiply and add (MAC).** Multiply each input code by its weight code, and add up the products in a wide register, the **accumulator**.
2. **Add the bias.** The bias is stored as an integer with the scale of the sum, $S_w S_x$, so it adds to the accumulator directly: $q_b = \mathrm{round}(b / (S_w S_x))$.
3. **Activation.** ReLU keeps the sums at or above 0. It works on the accumulator, before rounding.
4. **Requantize.** The sum has the tiny scale $S_w S_x$. The next layer wants 8-bit codes with its own scale $S_y$. [Requantization](requantization.md) rescales the sum with an integer multiplier and a shift, and rounds it into the output format.

The output codes go to the next layer, and the steps repeat.

## Why the accumulator is 32 bits

An 8-bit code times an 8-bit code needs up to 16 bits. A layer adds up many products: a 3×3 conv with 64 input channels adds 576 of them. Each added product can need more bits, about 1 bit for each doubling of the count. A 32-bit accumulator has room for more than 100,000 full-size products, which covers almost every layer. This is why INT8 inference uses INT32 sums and an INT32 bias.

## The last layer

The last layer often skips requantization, and returns its INT32 sums as the logits. They are exact, and finding the largest logit needs no scale.

## In Quaver

The integer mode of Quaver runs exactly these steps. [Inside a quantized layer](../guides/inside-a-layer.md) shows them in the code of `QuantConv2d`, and [Device math](../api/device-math.md) lists the exact integer steps for each op.
