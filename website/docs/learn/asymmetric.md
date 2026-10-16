---
title: "Asymmetric quantization"
description: "A zero point that moves the range to fit the data: no wasted codes, but extra integer math."
---

Asymmetric quantization maps the range of the data, from its smallest value to its largest, onto the full range of codes. The zero point is no longer 0. It moves to wherever the real number 0 falls.

## The rule

$$
S = \frac{r_{max} - r_{min}}{q_{max} - q_{min}}, \qquad Z = \mathrm{round}\left(q_{min} - \frac{r_{min}}{S}\right)
$$

Quantize and dequantize as usual:

$$
q = \mathrm{clip}\left(\mathrm{round}\left(\frac{r}{S}\right) + Z,\ q_{min},\ q_{max}\right), \qquad \hat{r} = S\,(q - Z)
$$

## The same example

Take the data from −400 to 1000 of the [symmetric page](symmetric.md), in INT8:

- $S = 1400 / 255 \approx 5.49$
- $Z = \mathrm{round}(-128 + 400 / 5.49) = \mathrm{round}(-55.1) = -55$
- −400 becomes $\mathrm{round}(-72.9) - 55 = -128$, and 1000 becomes $\mathrm{round}(182.1) - 55 = 127$.

Every code is now used. The step is 5.49 instead of 7.87, so the rounding error is about 30% smaller.

## What it costs

The zero point must be taken out before two codes are multiplied. A layer with an asymmetric input $x$ and symmetric weights $w$ computes:

$$
\sum_i q_{w,i}\,(q_{x,i} - Z_x) = \sum_i q_{w,i}\, q_{x,i} \;-\; Z_x \sum_i q_{w,i}
$$

The second term depends only on the weights and $Z_x$, so it is a constant for each output channel. A device can add it to the bias ahead of time, and pay nothing at run time.

If the weights are asymmetric too, one more term appears:

$$
\sum_i (q_{w,i} - Z_w)(q_{x,i} - Z_x) = \sum_i q_{w,i}\, q_{x,i} - Z_x \sum_i q_{w,i} - Z_w \sum_i q_{x,i} + n\,Z_w Z_x
$$

The term $Z_w \sum_i q_{x,i}$ depends on the input, so the device must compute it for every input. This is why weights are usually symmetric, even when activations are not.

## When to use it

- Activations that are not centered on 0, and can be negative: for example, the output of a layer without ReLU, or after GELU.
- Very low bit widths (4 bits or less), where every code counts.

When the data is never negative, as after ReLU, an [unsigned symmetric format](symmetric.md#data-that-is-never-negative) gives the same benefit without a zero point.

## In Quaver

Pass `symmetric=False` to a scaler to give an INT format a zero point. FP formats are always symmetric. The `_asym` models of the [ResNet-8 example](../examples/resnet8.md) use zero points on every weight and activation.
