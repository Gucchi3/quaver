---
title: "Block scales (NVFP4)"
description: "Use NVFP4 and other block-scaled formats in a model: block-scaled convs, chaining them, W4A8, and integer block scales."
---

Block-scaled formats such as NVFP4 keep 4-bit values accurate by giving every 16 values their own small scale. [Block scales](../learn/block-scales.md) explains how they work. This page shows how to use them in a model.

## The layers

`QuantBlockConv2d` and `QuantBlockLinear` are the block-scaled versions of `QuantConv2d` and `QuantLinear`. Their weights are always block-scaled. Three settings decide what goes in and out:

| Setting | Choices |
|---|---|
| `input_format` | `"ufp4"`, `"fp4"`, … quantizes the input to blocks inside the layer. `None` takes the input as it is: blocks from the layer before, or INT8 |
| `output_format` with `output_block` | `output_block=True` gives blocks in `output_format`, for a next block-scaled layer. `False` requantizes to `output_format` with one scale, such as INT8 or UINT8 |
| `scale_format` | `"ue4m3"` (NVFP4), or `"uint6"` / `"uint10"` (integer block scales) |

The input channels must be a multiple of 16 (the block size), and so must the output channels when `output_block=True`.

:::tip Use uint6 block scales

For integer devices, **uint6 is the block-scale format to use.** Its sums fit in 32 bits ([The width of the accumulator](#the-width-of-the-accumulator)). uint10 and ue4m3 (NVFP4) need a wider accumulator: 64-bit adds on a 32-bit CPU, or a wide accumulator in your hardware. Choose them only if your device has one.

:::

## A block of two NVFP4 convs

The first conv takes UINT8 codes and turns them into UFP4 blocks. It gives blocks (after ReLU) to the second conv, which requantizes its output to UINT8 for the next layer:

```python
from torch import nn

from quaver import QuantBlockConv2d, QuantConv2d, Quantizer, RangeScaler, build_format


class Nvfp4Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # conv1: quantize its UINT8 input to UFP4 blocks, and give UFP4 blocks (after ReLU) to conv2
        self.conv1 = QuantBlockConv2d(channels, channels, 3, padding=1, input_format="ufp4", output_format="ufp4", output_block=True, activation="relu")
        # conv2: take the blocks as they are, and requantize to UINT8 for the next layer
        self.conv2 = QuantBlockConv2d(channels, channels, 3, padding=1, output_format="uint8", activation="relu")

    def forward(self, x, x_params):
        return self.conv2(*self.conv1(x, x_params))


class Nvfp4Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantizer    = Quantizer(build_format("int8"), RangeScaler(), batch_axis=0)
        self.stem         = QuantConv2d(3, 32, 3, padding=1, output_format="uint8", activation="relu")
        self.block        = Nvfp4Block(32)
        self.integer_mode = False

    def forward(self, images):
        x, params = self.quantizer.encode(images) if self.integer_mode else self.quantizer(images)
        x, params = self.stem(x, params)
        return self.block(x, params)
```

- The first layer stays INT8 (`QuantConv2d`). The input and the output layers are the most sensitive, so models usually keep them at 8 bits.
- After ReLU, the values are not negative, so `ufp4` (0 to 7, one more mantissa bit) fits better than `fp4`.
- The scale of each input block is found at run time, from the codes, with integer math. The scale of the whole tensor is fixed after training, like any activation scale.

The NVFP4 model of the [ResNet-8 example](../examples/resnet8.md) is built this way, with a residual add in each block, and reaches 81.71% (the float model: 81.86%).

## W4A8: block-scaled weights, INT8 activations

With `input_format=None` and INT8 inputs, the weights are NVFP4 and the activations stay INT8. This is a common choice for language models:

```python
from quaver import QuantBlockLinear

linear = QuantBlockLinear(64, 32, output_format="int8")   # fp4 + ue4m3 weights, INT8 in, INT8 out
```

## Integer block scales

With `scale_format="uint6"` or `"uint10"`, the block scales are plain integers. A device then needs no FP math for the scales at all:

```python
conv = QuantBlockConv2d(32, 16, 3, padding=1, weight_format="int4", scale_format="uint6", input_format="uint4", output_format="int8")
```

## The width of the accumulator

:::tip Use uint6 block scales

For integer devices, **uint6 is the block-scale format to use.** With uint6, the sums of a layer fit in 32 bits, so a 32-bit CPU adds them with one instruction. uint10 and ue4m3 (NVFP4) need a wider accumulator: 64-bit adds on a 32-bit CPU, or a wide accumulator in your hardware. Choose them only if your device has one.

:::

A block-scaled layer adds up products of (element code × block-scale code) on both sides. The block-scale codes make these integers much larger than plain INT8 codes, so the sums can need more than 32 bits. On a 32-bit CPU, every add of a wider sum takes two instructions instead of one, and the layer runs much slower. Check the width before you choose a format for such a device.

The worst case is about:

$$
\text{bits} = \lceil \log_2(\max|x| \cdot \max|w| \cdot K) \rceil + 1
$$

where $\max|x|$ and $\max|w|$ are the largest integers of the two sides (element code × block-scale code), and $K$ is the number of products in one sum (input channels × kernel area for a conv).

| Weights, activations, block scales | Worst case, K = 576 (64 channels, 3×3) | Worst case, K = 4096 | Measured, 64 channels, 3×3 | On a 32-bit CPU |
|---|---|---|---|---|
| fp4, ufp4, ue4m3 (NVFP4) | 55 bits | 58 bits | 50 bits | needs 64-bit adds |
| int4, uint4, uint10 | 38 bits | 40 bits | 33 bits | over 32 bits |
| int4, uint4, uint6 | 30 bits | 32 bits | 25 bits | fits in 32 bits |
| fp4, ufp4, uint6 | 31 bits | 34 bits | | fits up to K = 1024 |
| fp4 + ue4m3, int8 (W4A8, NVFP4 weights) | 39 bits | 42 bits | 34 bits | over 32 bits |
| fp4 + uint6, int8 (W4A8) | 27 bits | 30 bits | 22 bits | fits in 32 bits |
| int4 + uint6, int8 (W4A8) | 27 bits | 29 bits | 21 bits | fits in 32 bits |
| int8, int8 (no block scales) | 25 bits | 28 bits | | fits in 32 bits |

NVFP4 is large because its E4M3 block scales span about $2^{18}$ (from $2^{-9}$ to 448), and a product carries two of them. GPUs avoid this by adding the products in FP32. An integer device that wants the exact NVFP4 sums needs a wide accumulator.

So use integer block scales of 6 bits (`scale_format="uint6"`). In the ResNet-8 test, int4 with uint6 block scales reached 81.68%, against 81.83% for NVFP4, and its sums fit in 32 bits. The export writes the accumulator and the bias of a layer as int32 when its worst case fits in 32 bits, and as a wide integer when it does not.

## Things to know

- **Training is slower.** The training mode of these layers also runs the integer math, since the sums can be too wide for float values. They are slower than the other layers.
- **The modes still match.** As with every layer, the training mode and the integer mode give the same codes.
