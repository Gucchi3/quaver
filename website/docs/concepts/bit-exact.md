---
title: "How the modes match bit for bit"
description: "Why the training mode gives the same integer codes as the integer mode and your device, and what you need for it."
---

In Quaver, the model you train and the model your device runs give the same integer codes. So the accuracy you see after QAT is the accuracy of your device, and the outputs of the integer mode can serve as test vectors for your hardware.

## Why normal fake quantization is not enough

Normal [fake quantization](../learn/qat.md) does not match a device that uses integer math. In a test with one 3×3 conv layer (about 400,000 outputs), 0.046% of the outputs were off by one code. There are two reasons.

- **The scale ratio.** A device cannot multiply by a float such as "input scale × weight scale ÷ output scale". It uses an integer multiplier and a shift instead, which is a little off from the real ratio. Fake quantization uses the real ratio.
- **Float error.** Fake quantization adds up the products in float, so the sums have small errors. Near the middle between two codes, a small error can change the rounding.

## What Quaver does

In the training mode, each Quaver layer runs the same integer math as the integer mode:

1. The conv or linear layer adds up its products in float, as usual.
2. Quaver rounds each sum to the nearest integer. The float error is much smaller than 0.5, so this gives the exact integer sum (the INT32 sum that your device gets).
3. Every later step (the bias, the activation, and the requantization with a multiplier and a shift) uses the same integer functions as the integer mode.
4. For the backward pass, Quaver attaches the gradients of the float math ([STE](../learn/qat.md)).

So the two modes share every step that could round differently. They do not just come close. They give the same codes by design.

## What you need for a match

| Condition | Why |
|---|---|
| TF32 is off (`precision_disable_tf32()`) | TF32 keeps only 10 bits of the mantissa, so the float sums are too far off to round back to the exact integer sums. |
| The scales are fixed (`eval()`) | If a scale changes between the two runs, the codes change too. |
| Your device uses the same requantization | Quaver uses a 16-bit multiplier and a shift by default (`"mul16_shift"`), or a shift only (`"shift_only"`). For another way, such as the 31-bit multiplier of CMSIS-NN, add it to `REQUANTIZE_REGISTRY`. |
| The sums fit in INT32 | Quaver does not handle overflow, because a device does not overflow in normal use either. |
| Each layer requantizes its own output | If you quantize a sum with a separate `Quantizer` instead, the scale ratio is not a multiplier and a shift, and the codes can be one off. |

The training mode can also use float32 math to train faster (see [Fast training](fast-training.md)). Then a code can be one off in rare places during training, but tests still run in float64 and match exactly.

## How it was checked

- Each op was checked against a slow reference written with Python integers and fractions.
- For the whole model, the training mode and the integer mode were compared on the same inputs.
- ResNet-8 was also written again as a separate program that uses only integer math (no float at all), with the constants from Quaver. On the 10,000 test images, every code of every layer, and the INT32 logits, matched for all five quantized models. The same check passed for a model with SimA attention (20 settings), and for the NVFP4 model.

[Device math](../api/device-math.md) lists each step that your device must run, such as the rounding of the requantization.
