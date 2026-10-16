---
title: "Granularity"
description: "How many scales a tensor has: one for the whole tensor, one for each channel or token, or one for each small block."
---

A tensor can share one scale, or have many. Granularity says how many values share one scale.

| Granularity | Number of scales | Common use |
|---|---|---|
| Per tensor | 1 | activations |
| Per channel | one for each output channel | weights |
| Per token | one for each token | activations of transformers |
| Per group (per block) | one for each small group of values, such as 16 or 32 | 4-bit formats, such as NVFP4 |

## Why more scales help

One scale must cover the widest part of the tensor. If one channel has values up to 10 and another only up to 0.1, a shared scale fits the first channel, and the second channel gets only a few codes. It is almost all rounding error.

With one scale for each channel, each channel uses all its codes. This is why weights are almost always quantized per output channel: the channels of a trained layer often differ a lot in size.

## What it costs

- **Memory.** Each scale must be stored. For per-channel weights this is tiny. For groups of 16 values, it adds a scale for every 16 numbers. Formats like NVFP4 store these scales in 8 bits to keep the cost small.
- **Integer math.** A device turns each scale into a multiplier and a shift for [requantization](requantization.md). Per-channel scales need a small table of multipliers and shifts, one for each output channel.

## Which axis

The scales of a matrix product must not change along the axis that is summed. If they did, the terms of the sum would have different scales, and could not be added as integers.

- Weights: one scale for each **output** channel. Each output sums over the input channels, and all of them share the scale of that output.
- Activations: one scale for each **token** (row) is fine, because each output sums over the channels of one token.

Groups along the summed axis need special care. [Block scales](block-scales.md) shows how NVFP4 does it.

## In Quaver

```python
Granularity()                                   # per tensor
Granularity("channel", axis=0)                  # per channel (axis 0 of a weight is the output channel)
Granularity("block", axis=-1, block_size=16)    # per block of 16 along the last axis
```

One block over the whole last axis gives one scale for each token.
