---
title: "Block scales"
description: "One small scale for every block of 16 values: how NVFP4 keeps 4-bit values accurate, and how a device can compute it with integers."
---

At 4 bits, there are only 16 codes. One scale for a whole tensor, or even for a whole channel, is too coarse: a few large values set the scale, and the small values round to 0. Block scales fix this. Each small block of values, such as 16 in a row, gets its own scale.

## NVFP4

NVFP4 is a block format from NVIDIA. It has three parts:

| Part | Format | How many |
|---|---|---|
| Elements | FP4 (E2M1): 0, 0.5, 1, 1.5, 2, 3, 4, 6, and the same with a minus sign | every value |
| Block scale | E4M3, an 8-bit float with no sign (up to 448) | one for each block of 16 values |
| Tensor scale | one number (FP32 on a GPU) | one for the whole tensor |

A real value is the element times its block scale times the tensor scale:

$$
r \approx x \cdot s_{block} \cdot t
$$

Quantizing a block goes like this:

1. The tensor scale fits the largest value of the tensor: $t = \max|r| / (6 \cdot 448)$.
2. Each block scale fits the largest value of its block, rounded to E4M3: $s_{block} = \mathrm{E4M3}(\max_{block}|r| / 6 / t)$.
3. Each element is rounded to FP4: $x = \mathrm{FP4}(r / (s_{block} \cdot t))$.

So in every block, the largest value lands near 6, the top of FP4, and the small values get fine steps. The cost is 8 extra bits for every 16 values, or 0.5 bits for each value.

Other block formats work the same way, with other parts. MXFP4 uses blocks of 32 and a power-of-two scale. Quaver also supports integer block scales (uint6 and uint10), and INT8 elements with block scales (for example, 4-bit weights and 8-bit activations).

## Why the blocks run along the summed axis

In a matrix product, each output adds up products along one axis (the input channels). The blocks run along that same axis. So each block of 16 products has one weight block scale and one activation block scale, and the device can add up the 16 products first, and then multiply the sum by the two scales once.

## Integer math on a device

GPUs add up the blocks in FP32. A device with integer hardware can do it exactly with integers instead:

1. **Elements.** Each FP4 value times 2 is a small integer: 0, ±1, ±2, ±3, ±4, ±6, ±8, ±12. The product of two of them is at most 144, and the sum of 16 products fits in 13 bits.
2. **Block scales.** Each E4M3 scale is an integer mantissa $M$ (up to 15) times a power of two: $M \cdot 2^E$. The product of two block scales is $(M_a M_w) \cdot 2^{E_a + E_w}$.
3. **One block.** Multiply the block sum by $M_a M_w$, and shift it left by $E_a + E_w$ plus a fixed offset. This is an integer.
4. **All the blocks.** Add the block values in one wide integer accumulator, with no rounding. For 1024 inputs, 54 bits are enough.
5. **Output.** The tensor scales are fixed, so they go into the multiplier and shift of a normal [requantization](requantization.md), with one rounding at the end.

Since nothing is rounded before the end, the result does not depend on the order of the sums. A device and Quaver get exactly the same codes. The price is the width: exact NVFP4 sums need about 50 bits or more, which is slow on a 32-bit CPU. Integer block scales of 6 bits keep the sums within 32 bits, so they are the ones to use on integer devices; 10-bit integer block scales need a wider accumulator too. [Block scales (NVFP4)](../guides/block-scale.md#the-width-of-the-accumulator) compares the formats.

## Activations in NVFP4

The weights are quantized once, after training. The activations must be turned into NVFP4 at run time, for each input: find the largest value of each block, round the block scale to E4M3, and round each element to FP4. A device can do all of this with integer multiplies, comparisons, and shifts, with no division. The tensor scale of the activations is fixed ahead of time, like any [static scale](static-dynamic.md).

## In Quaver

`BlockScaler("ue4m3")` gives NVFP4 block scales, and `QuantBlockConv2d` and `QuantBlockLinear` run block-scaled layers with integer math in both modes. The NVFP4 model of the [ResNet-8 example](../examples/resnet8.md) keeps almost all the accuracy of the float model.
