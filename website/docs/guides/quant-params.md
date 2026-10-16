---
title: "Quantized values and QuantParams"
description: "How values, codes, scales, and zero points relate, and why Quaver passes them as a pair."
---

In Quaver, a quantized tensor always travels as a pair: the tensor and its `QuantParams`. This page shows what the pair holds, and how to move between values and integer codes.

## What QuantParams holds

| Field | Meaning |
|---|---|
| `scale` | The [scale](../learn/scale-zero-point.md): one number, or one for each channel or block |
| `zero_point` | The code that stands for 0. It is 0 for symmetric scales |
| `number_format` | The format of the codes, such as int8 or fp4 |

A code and its real value are tied by this rule:

```text
real value = (code - zero_point) × code_scale
code_scale = scale × step_min of the format
```

For INT formats, `step_min` is 1, so `code_scale` is the scale. For FP formats, the codes count in steps of the smallest value of the format (0.5 for fp4), so that they are integers too.

## An example

Here, one tensor is quantized in three ways.

```python
import torch

from quaver import DynamicScaler, Granularity, Quantizer, build_format

x = torch.tensor([[-1.0, -0.25, 0.1, 0.5, 2.0]])

for name, symmetric in (("int8", True), ("fp4", True), ("uint8", False)):
    quantizer      = Quantizer(build_format(name), DynamicScaler(symmetric=symmetric), Granularity())
    values, params = quantizer(x)
    codes, _       = quantizer.encode(x)
    print(name, values, codes, params.scale, params.zero_point)
```

| Format | Values (fake-quantized) | Codes | Scale | Zero point |
|---|---|---|---|---|
| int8 | -1.0079, -0.2520, 0.0945, 0.5039, 2.0000 | -64, -16, 6, 32, 127 | 0.0157 (= 2 / 127) | 0 |
| fp4 | -1.0000, -0.3333, 0.1667, 0.5000, 2.0000 | -6, -2, 1, 3, 12 | 0.3333 (= 2 / 6) | 0 |
| uint8 (asymmetric) | -1.0000, -0.2471, 0.1059, 0.5059, 2.0000 | 0, 64, 94, 128, 255 | 0.0118 (= 3 / 255) | 85 |

For fp4, the code 12 means 12 × 0.3333 × 0.5 = 2.0.

## Move between values and codes

| Function | From | To |
|---|---|---|
| `quantizer(x)` | float values | fake-quantized values and QuantParams |
| `quantizer.encode(x)` | float values | codes (int64) and QuantParams |
| `params_codes(values, params)` | fake-quantized values | codes |
| `params_values(codes, params)` | codes | real values |

`params_codes()` and `params_values()` are exact: they give back the same codes and values, with no rounding error.

## Why a pair

A quantized layer needs the scale of its input to work. In the integer mode, it multiplies the input scale and the weight scale to find the scale of its sums, and from that, the multiplier and shift of its requantization. So every Quaver layer takes a pair and returns a pair:

```python
x, params = quantizer(images)
x, params = conv1(x, params)
x, params = conv2(x, params)
```

The pair works the same way in both modes. In the training mode, `x` holds fake-quantized values. In the integer mode, it holds codes. The QuantParams are the same in both.
