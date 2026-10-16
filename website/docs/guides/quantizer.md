---
title: "Write a quantizer"
description: "Make a quantizer from a number format, a scaler, and a granularity, or build it from settings."
---

This page explains the first part of [the program](overview.md): the quantizer of the input. A quantizer turns float values into quantized values. In Quaver, you build it from three parts.

```python
from quaver import Granularity, Quantizer, RangeScaler, build_format

# The input quantizer of the model: INT8, one scale for the whole tensor, set from the range of the values
quantizer = Quantizer(build_format("int8"), RangeScaler(), Granularity(), batch_axis=0)
```

| Part | What it decides |
|---|---|
| Number format | Which values the codes can take |
| Scaler | How the scale (and the zero point) is set |
| Granularity | How many scales the tensor has |

## Number formats

`build_format()` returns a format by its name.

| Name | Format |
|---|---|
| `"int8"`, `"int4"`, `"int2"` | signed [integers](../learn/int-fp.md) |
| `"uint8"`, `"uint4"`, `"uint2"` | unsigned integers, for example after a ReLU |
| `"fp4"` | 4-bit float (E2M1), the element format of NVFP4 |
| `"ufp4"` | unsigned 4-bit float (E2M2) |

## Scalers

The scaler sets the scale. Choose it by how the values change.

| Scaler | How it sets the scale | Use it for |
|---|---|---|
| `DynamicScaler()` | Measures the range of each tensor, every time | weights (the sample layers use it) |
| `RangeScaler()` | Keeps a [moving average](../learn/static-dynamic.md) of the range during training, and uses it after `eval()` | activations |
| `LsqScaler()` | Learns the scale with the loss ([LSQ](../learn/qat.md)) | weights and activations, when you want the best scale |
| `SharedScaler()` | One range for several tensors | [the two inputs of a residual add](../learn/shared-scales.md) |
| `BlockScaler()` | One small scale for each block, and one for the tensor | [NVFP4 and other block formats](../learn/block-scales.md) |

Pass `symmetric=False` to give an INT format a [zero point](../learn/asymmetric.md). With `LsqScaler`, this gives LSQ+.

## Granularity

`Granularity` decides how many scales there are ([Granularity](../learn/granularity.md)).

```python
Granularity()                                   # one scale for the whole tensor
Granularity("channel", axis=0)                  # one scale for each index of axis 0, such as each output channel of a weight
Granularity("block", axis=-1, block_size=16)    # one scale for each block of 16 values along the last axis
```

## The batch axis

Give `batch_axis=0` to the quantizers of activations (the inputs and outputs of layers). It tells the quantizer which axis holds the samples of a batch. LSQ uses it to scale its gradient for one sample. Weights have no batch axis, so leave it out for them.

## Use the quantizer

A quantizer returns a pair: the quantized tensor and its QuantParams (the scale, the zero point, and the format).

```python
values, params = quantizer(images)        # training mode: fake-quantized values (float)
codes, params  = quantizer.encode(images) # integer mode: integer codes (int64)
```

The next page explains this pair.

## Build a quantizer from settings

To set quantizers from a config file, fill a `QuantizerSpec` and pass it to `build_quantizer()`.

```python
from quaver import QuantizerSpec, build_quantizer

settings  = {"format": "int8", "scaler": "percentile", "percentile": 99.9, "batch_axis": 0}
quantizer = build_quantizer(QuantizerSpec(**settings))
```

| `scaler` | The scaler it builds |
|---|---|
| `"dynamic"` | `DynamicScaler` |
| `"max"` | `RangeScaler` that follows the maximum |
| `"percentile"` | `RangeScaler` that follows a percentile (99.9 by default), to ignore rare large values |
| `"lsq"` | `LsqScaler` (LSQ+ when `symmetric` is false) |
| `"shared"` | a new `SharedScaler` (to share it, make one and pass it to each quantizer) |
| `"nvfp4"` | `BlockScaler("ue4m3")` |

`QuantizerSpec` also takes `granularity`, `axis`, `block_size`, `symmetric`, `round_mode`, `momentum`, `pow2` (scales that are powers of two, for a device that can only shift), `lsq_init`, and `freeze_step`. An unknown scaler name raises an error that lists the names you can use. To add your own scaler, add a builder to `SCALER_REGISTRY`.
