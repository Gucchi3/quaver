---
title: "Supported combinations"
description: "Which formats, rounding rules, and scales Quaver supports, how they combine, and which layers support each one."
sidebar_position: 3
---

What Quaver supports, and how it fits together: the number formats, the pairs of weights and activations, the rounding rules, the scalers, and the layers.

## Number formats

| Name | What it is | Bits | Range | Weights | Activations | Block elements | Block scales |
|---|---|---|---|---|---|---|---|
| `int2`, `int4`, `int8` | signed integers | 2, 4, 8 | $-2^{b-1}$ to $2^{b-1} - 1$ | yes (symmetric or with a zero point) | yes (symmetric or with a zero point) | yes | |
| `uint2`, `uint4`, `uint8` | unsigned integers | 2, 4, 8 | 0 to $2^b - 1$ | yes | yes (after ReLU) | yes | |
| `uint6`, `uint10` | unsigned integers | 6, 10 | 0 to $2^b - 1$ | | | | yes (integer block scales; use uint6 on integer devices, since uint10 needs a wider accumulator) |
| `fp4` | E2M1 | 4 | ±6 | yes (symmetric) | yes (symmetric) | yes (NVFP4) | |
| `ufp4` | E2M2, unsigned | 4 | 0 to 7 | yes | yes (after ReLU) | yes | |
| `ue4m3` | E4M3, unsigned | 7 (stored in 8) | $2^{-9}$ to 448 | | | | yes (NVFP4) |

- Only INT formats can have a zero point. Giving an FP format an asymmetric scaler raises `ValueError`.
- A block format is an element format, a block-scale format, and a block size: NVFP4 is fp4 + ue4m3 per 16. After a ReLU, ufp4 + ue4m3 fits better. Integer block scales use uint6 or uint10.
- The accumulator and the bias are INT32. With a block-scaled tensor in the sum, they can need up to 62 bits (exact in int64): about 55 bits for NVFP4, 38 for uint10 block scales, and 30 for uint6 block scales, for a 64-channel 3×3 conv. [The width of the accumulator](../guides/block-scale.md#the-width-of-the-accumulator) has the table.

## Weights and activations

The format of the weights and of the activations can be chosen for each layer. These are the pairs of the examples:

| Pair | Weights | Activations | Accumulator | Layers | ResNet-8 on CIFAR-10 |
|---|---|---|---|---|---|
| W8A8 | int8 (per channel) | uint8 | INT32 | `QuantConv2d`, `QuantLinear` | 82.00% |
| W8A8 with zero points | int8 | int8 | INT32 | the same, with `weight_symmetric=False`, `output_symmetric=False` | 82.04% |
| W4A4 | int4 | uint4 | INT32 | the same | 78.51% |
| W4A4 with zero points | int4 | int4 | INT32 | the same | 78.59% |
| W4A4 (FP) | fp4 | ufp4 | INT32 | the same | 80.50% |
| NVFP4 | fp4 + ue4m3 per 16 | ufp4 + ue4m3 per 16 | 62 bits | `QuantBlockConv2d`, `QuantBlockLinear` | 81.71% |
| W4A8 | fp4 + ue4m3 per 16 | int8 (one fixed scale) | 62 bits | `QuantBlockLinear` with `input_format=None` | |
| Integer block scales | int4 or fp4 + uint6 / uint10 per 16 | ufp4 or uint4 + uint6 / uint10 per 16 | 62 bits | `QuantBlockConv2d`, `QuantBlockLinear` | |
| W2A2 | int2 | uint2 or int2 | INT32 | `QuantConv2d`, `QuantLinear` | |

The float model reaches 81.86%. Even in the 4-bit models, the input layer, the output layer, and the residuals stay at 8 bits. The [CNN + ViT example](../examples/cnn-vit.md) gives the results with SimA attention.

## Rounding

| Where | Default | How to change it |
|---|---|---|
| Quantizing weights and inputs (`Quantizer`) | INT: half away from zero; FP: half to even | `round_mode`: `"half_away"`, `"half_even"`, `"half_up"`, or `"floor"` |
| The bias | half away from zero | the `mode` of `bias_quantize()` |
| Requantization to an INT format | `"mul16_shift"` and `"shift_only"`: half up; `"exact"`: half away | `style` (add a style to `REQUANTIZE_REGISTRY`) |
| The residual add | the rounding of the style, once after the sum | `style` |
| Average pooling | the usual rounding of the input format | `round_mode` of `QuantAvgPool2d` |
| Any output on an FP grid | always half to even, like FP hardware | fixed |
| Block scales: block codes and elements | half to even | fixed |
| SimA: the multipliers found at run time, NEQ codes | half up | fixed |

The rounding always comes before the zero point is added, as in TFLite and on devices.

## Scalers

| Scaler (name in `QuantizerSpec`) | How the scale is set | Fixed or measured | Granularity | Zero point | `pow2` | Main use |
|---|---|---|---|---|---|---|
| `DynamicScaler` (`"dynamic"`) | the maximum (or a percentile) of each tensor, every time | measured | tensor, channel, block, token | both | yes | weights |
| `RangeScaler` (`"max"`) | a moving average of the maximum while training | fixed after training | tensor, channel | both | yes | activations |
| `RangeScaler` (`"percentile"`) | a moving average of the 99.9th percentile | fixed after training | tensor, channel | both | yes | activations |
| `LsqScaler` (`"lsq"`) | learned by the loss (LSQ+ with a zero point) | fixed after training | tensor, channel | both | yes | weights and activations in QAT |
| `SharedScaler` (`"shared"`) | one scale for several tensors | fixed after training | tensor | both | yes | the residual stream of a block |
| `BlockScaler` (`"nvfp4"`) | each block from its largest value, rounded to the block-scale format | blocks measured; tensor measured or fixed | blocks of 16 | symmetric | no | NVFP4, integer block scales |

On a device, the only scales measured at run time with integer math are those of NEQ in SimA and the block scales of activations. The scales of weights are found once, by Python, so they can be measured or fixed.

## Layers and ops

| Layer or op | Weights | Input | Output | Notes |
|---|---|---|---|---|
| `QuantConv2d`, `QuantLinear` | any format, per channel | codes with one scale, any format | requantized to `output_format`, or the INT32 accumulator | `activation`, `style`, `output_scaler` |
| `QuantBlockConv2d`, `QuantBlockLinear` | element + block scale per 16 | quantized to blocks here (`input_format`), or taken as it is | block-scaled (`output_block=True`), or requantized | 62-bit accumulator; the training mode also runs the integer math |
| `QuantAdd` | | two inputs, any format | `output_format` | rounded once; with one shared scale, a plain integer add |
| `QuantAvgPool2d` | | codes, any format | the same format and scale | `round_mode` |
| `QuantSimaAttention` | `weight_format` (int4 by default) | codes with one scale | the INT32 accumulator of the projection, or `output_format` | NEQ per tensor or per token; `act_format` of 8 bits or less |

Every layer runs in the training mode, the integer mode, and the float mode, and its training and integer modes run the same integer math.

## Limits by design

| Combination | Why |
|---|---|
| A zero point on an FP format | FP formats have no zero point (`ValueError`) |
| `pow2` with block scales | Block scales sit on the grid of their own format (`ValueError`) |
| A zero point with block scales | The zero point of `BlockScaler` is always 0 |
| Another requantization of a device, such as CMSIS-NN's | Add its style to `REQUANTIZE_REGISTRY` |
