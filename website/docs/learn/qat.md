---
title: "QAT, STE, and LSQ"
description: "Train with the rounding in the loop: fake quantization, gradients through rounding, and learned scales."
---

Quantization-aware training (QAT) trains the model with the quantization in the loop. The model sees the rounding and clipping errors during training, and its weights learn to work well with them. It recovers most of the accuracy that [PTQ](ptq.md) loses at low bit widths.

## Fake quantization

QAT runs on a normal GPU in float. Each quantized tensor is quantized and right away dequantized:

$$
\hat{r} = S \cdot \left(\mathrm{clip}\left(\mathrm{round}\left(\frac{r}{S}\right) + Z,\ q_{min},\ q_{max}\right) - Z\right)
$$

$\hat{r}$ is a float, but it can only take the values of the quantized format. This is called **fake quantization**. The layers then compute with these values in float, as usual.

It is usual to train the model in float first, and then run QAT for a few epochs, starting from the float weights.

## STE: gradients through rounding

Rounding is a staircase: flat almost everywhere, so its gradient is 0, and nothing could learn. The **straight-through estimator** (STE) fixes this. In the backward pass, it treats rounding as if it were the identity (a slope of 1) inside the range, and as flat (a slope of 0) outside it, where the value is clipped:

$$
\frac{\partial \hat{r}}{\partial r} \approx
\begin{cases}
1 & \text{inside the range} \\
0 & \text{outside (clipped)}
\end{cases}
$$

This is not the true gradient, but it works well in practice.

## LSQ: learn the scale

Learned step size quantization (LSQ) makes the scale $S$ a parameter, and trains it with the loss, like a weight. The gradient of $\hat{r}$ with respect to $S$ (with the STE) is:

$$
\frac{\partial \hat{r}}{\partial S} =
\begin{cases}
\mathrm{round}(r/S) - r/S & \text{inside the range} \\
q_{min} \text{ or } q_{max} & \text{clipped below or above}
\end{cases}
$$

Inside the range, the gradient pushes $S$ to cut the rounding error. Outside, it pushes $S$ to widen the range. The balance is the one that is best for the loss. LSQ also scales this gradient down, so that the scale learns at a speed similar to the weights. LSQ+ adds a learned zero point for asymmetric data.

## Why Quaver does more than fake quantization

Fake quantization rounds each tensor, but the float math between them is not the integer math of a device. So a model trained with it can give slightly different codes on the device. Quaver's training mode runs the same integer math as the device, so the codes match bit for bit. See [How the modes match bit for bit](../concepts/bit-exact.md).
