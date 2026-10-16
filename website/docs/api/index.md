---
title: "API reference"
description: "The functions and classes of Quaver, one page per file: the basic API, the folders, and the symbols of the formulas."
sidebar_position: 1
---

Every public function and class of Quaver, one page for each file. Each entry shows how to call it, what it does, its arguments with their types, and an example.

## The basic API

The names you use most can be imported from the top of the package:

```python
from quaver import QuantConv2d, Quantizer, RangeScaler, build_format, switch_integer_mode
```

| Name | What it is |
|---|---|
| **Layers** | |
| [`QuantConv2d`](samples/quant_conv2d.md), [`QuantLinear`](samples/quant_linear.md) | Conv2d and Linear with quantized weights, an INT32 bias, and requantization |
| [`QuantBlockConv2d`](samples/quant_block_conv2d.md), [`QuantBlockLinear`](samples/quant_block_linear.md) | Block-scaled Conv2d and Linear (NVFP4 and the like) |
| [`QuantAdd`](samples/quant_add.md) | The residual add |
| [`QuantAvgPool2d`](samples/quant_avg_pool2d.md) | Average pooling |
| [`QuantSimaAttention`](samples/quant_sima_attention.md) | SimA attention without softmax |
| **Quantizers and scales** | |
| [`Quantizer`](quantizers/quantizer.md) | A format, a scaler, and a granularity put together |
| [`QuantizerSpec`](quantizers/types.md#quantizerspec), [`build_quantizer`](quantizers/build.md#build_quantizer) | Build a quantizer from settings |
| [`build_format`](formats/format.md#build_format) | A number format by name |
| [`DynamicScaler`](scales/dynamic_scaler.md), [`RangeScaler`](scales/range_scaler.md), [`LsqScaler`](scales/lsq_scaler.md), [`SharedScaler`](scales/shared_scaler.md), [`BlockScaler`](scales/block_scaler.md) | The scalers |
| [`Granularity`](scales/types.md#granularity) | How many scales a tensor has |
| [`QuantParams`](scales/types.md#quantparams) | The scale, zero point, and format of a quantized tensor |
| [`params_codes`](scales/params.md#params_codes), [`params_values`](scales/params.md#params_values) | Move between values and codes |
| **Whole model** | |
| [`switch_integer_mode`](model_tools/switch.md#switch_integer_mode), [`switch_quantizers`](model_tools/switch.md#switch_quantizers) | Switch the mode of every layer |
| [`calibrate_model`](model_tools/calibrate.md) | Measure the ranges before QAT, or for PTQ |
| [`freeze_scalers`](model_tools/freeze.md) | Stop the range updates |
| [`lsq_param_groups`](model_tools/lsq.md#lsq_param_groups), [`lsq_clamp_steps`](model_tools/lsq.md#lsq_clamp_steps) | Train LSQ scales |
| [`fold_batch_norms`](model_tools/fold.md) | Fold BatchNorm into the conv before it |
| [`precision_disable_tf32`](model_tools/precision.md#precision_disable_tf32), [`precision_set_math`](model_tools/precision.md#precision_set_math) | Float settings |
| [`model_export`](export/model_export.md) | Export `model.json` and `model.bin` for your device |

Every other name is imported from its folder, such as `from quaver.ops import requantize_params`.

## The folders

The folders build on each other from the bottom up. A folder only uses the folders above it in this list, never the ones below.

| Folder | What it holds | Uses |
|---|---|---|
| [formats/](formats/index.md) | Number formats, rounding, and the device ops | nothing |
| [scales/](scales/index.md) | `QuantParams`, granularity, and the scalers | formats |
| [quantizers/](quantizers/index.md) | `Quantizer`, and building it from settings | formats, scales |
| [ops/](ops/index.md) | The quantized ops, in a training, an integer, and a float version | formats, scales |
| [model_tools/](model_tools/index.md) | Whole-model helpers | quantizers, scales |
| [samples/](samples/index.md) | The sample layers | formats, scales, quantizers, ops |
| [export/](export/index.md) | `model_export()` for the device | formats, scales |

## Symbols in the formulas

| Symbol | Meaning |
|---|---|
| $r$, $\hat{r}$ | A real value, and its quantized value |
| $q$ | An integer code |
| $S$ | A scale |
| $Z$ | A zero point |
| $q_{min}$, $q_{max}$ | The smallest and largest codes of a format |
| step_min | The smallest step of a format: 1 for INT, 0.5 for fp4. A code of an FP format is its value ÷ step_min |
| code_scale | The real value of one step of the code: $S$ × step_min |
| $acc$ | An accumulator: a sum of products of codes (INT32, or 62 bits with block scales) |
| $M_0$, $n$ | The multiplier and the shift of a requantization: ratio ≈ $M_0 / 2^n$ |
| $p$ | The integer of NEQ that sets the scale of a normalized token |

[Supported combinations](combinations.md) lists which formats and scales work together, and [Device math](device-math.md) lists the integer steps of each layer.
