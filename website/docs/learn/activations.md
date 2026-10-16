---
title: "ReLU and ReLU6"
description: "ReLU and ReLU6 on integer codes are just clamps, and they can run on the sums before requantization."
---

ReLU and ReLU6 are the activations of most small models for devices. On integer codes, both are just clamps.

## ReLU

ReLU keeps positive values and turns negative ones into 0: $\mathrm{ReLU}(r) = \max(r, 0)$. A code $q$ with zero point $Z$ stands for $S\,(q - Z)$, which is negative when $q < Z$. So on codes:

$$
\mathrm{ReLU}(q) = \max(q, Z)
$$

With a symmetric scale ($Z = 0$), this is $\max(q, 0)$.

## ReLU6

ReLU6 also cuts values above 6: $\min(\max(r, 0), 6)$. The real number 6 is the code $\mathrm{round}(6 / S) + Z$, so:

$$
\mathrm{ReLU6}(q) = \min\big(\max(q, Z),\ \mathrm{round}(6 / S) + Z\big)
$$

The upper code is a constant, computed once from the fixed scale. If 6 is not exactly on a mark, the rounding makes the cut a little off from 6 in real terms, but training and the device use the same code, so they agree.

## Run it on the sums

In a conv or linear layer, the activation can run on the INT32 sums, before the [requantization](requantization.md):

- The sums are exact, so $\max(acc, 0)$ is exact too.
- The requantization clamps its output anyway. So a device can merge ReLU into it: just clamp from below at the zero point of the output instead of the bottom of the format.

This is why ReLU costs almost nothing on a device.

## Unsigned outputs

After ReLU, no value is negative. So the output can use an unsigned format, such as UINT8 or UINT4, with $Z = 0$, and keep all its codes for the real values ([Symmetric quantization](symmetric.md#data-that-is-never-negative)).

## In Quaver

Pass `activation="relu"` or `activation="relu6"` to a sample layer. The layer runs it on the sums, before the requantization, in both modes.
