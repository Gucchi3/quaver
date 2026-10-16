---
title: "Export to your device"
description: "Export the trained model as model.json and model.bin, choose the layout and the alignment, and read the files back."
---

The last step of [the program](overview.md) exports the model for your device. Quaver writes two files:

- `model.json` says what to compute: the layers in order, how they connect, their settings, and where each number is.
- `model.bin` holds every number, packed so that you can write it into the memory of your device (for example PSRAM) as it is.

[From Quaver to your device](../concepts/device-flow.mdx) explains the idea. This page shows how to use it.

## Export the model

```python
from quaver import model_export, switch_integer_mode

model.eval()                      # fix the scales
switch_integer_mode(model, True)  # export runs the model in the integer mode
model_export(model.cpu(), (images[:1].cpu(),), "tiny_net_export")
```

| Argument | Meaning |
|---|---|
| `model` | The trained model, in `eval()` and in the integer mode |
| `inputs` | A tuple of example inputs for `forward()`. Only their shape and type matter, so one image is enough |
| `folder` | The folder to write `model.json` and `model.bin` into |
| `alignment` | Each number array starts at a multiple of this many bytes (4 by default). Use the burst size of your memory, such as 32 |
| `layout` | The layout of each layer: `"channels_last"` (the default, also called HWC) or `"channels_first"` (CHW) |
| `json_values` | `True` (the default) also writes every number in `model.json`, in decimal. `False` leaves them out, to keep `model.json` small |

`model_export()` also returns the content of `model.json` as a dict. It needs numpy.

`forward()` of the model can return a tensor, or a pair of codes and QuantParams. With the pair, as in TinyNet, `model.json` also records the scale and the zero point of the output.

## What model.json holds

The top of `model.json` gives the size of `model.bin`, the alignment, the byte order (little), and the inputs and outputs of the model. Then `"layers"` lists what to run, in order. For TinyNet:

```text
quaver.quantize        images → quantize       the input image to int8 codes
QuantConv2d            conv1
QuantDepthwiseConv2d   depthwise2              your own layer, with its own class name
QuantConv2d            pointwise2
QuantDepthwiseConv2d   depthwise3
QuantConv2d            pointwise3
QuantAvgPool2d         pool
aten.flatten           pool.output → flatten
QuantLinear            classifier
```

Each layer has its `name` and `class`, its `inputs` and `outputs` (by name, with format, shape, scale, and zero point), its `settings`, and its `data`. Here is `depthwise2`:

```json
{
  "name": "depthwise2",
  "class": "QuantDepthwiseConv2d",
  "inputs":  [{"name": "conv1.output", "format": "uint8", "shape": [1, 32, 32, 32], "axes": ["batch", "h", "w", "channel"]}],
  "outputs": [{"name": "depthwise2.output", "format": "uint8", "shape": [1, 32, 16, 16], "scale": 0.0102, "zero_point": 0}],
  "settings": {"stride": 2, "padding": 1, "groups": 32, "activation": "relu", "kernel_size": [3, 3],
               "weight_format": "int8", "output_format": "uint8", "output_min": 0, "output_max": 255},
  "data": [
    {"role": "weight", "format": "int8", "shape": [32, 1, 3, 3], "stored_shape": [32, 3, 3, 1],
     "axes": ["out", "h", "w", "in"], "offset": 1396, "size": 288},
    {"role": "multiplier", "format": "uint16", "stored_shape": [32], "offset": 2100, "size": 64}
  ]
}
```

- **The settings** come from the attributes of the layer with known names: `stride`, `padding`, `groups`, `activation`, and so on. This is why `QuantDepthwiseConv2d` keeps its group count in `self.groups`.
- **The data** of a layer is named by role. The roles come from the ops the layer uses, so your own layer gets the same names as the sample layers: `weight`, `weight_zero_point`, `bias`, `multiplier`, `shift`, `input_zero_point`, `output_zero_point`, and so on.
- **`shape`** is the shape in PyTorch. **`stored_shape`** and **`axes`** say how the values are laid out in `model.bin`.
- **`offset`** and **`size`** say where the values are in `model.bin`, in bytes.
- **`values`** shows the same numbers in decimal, in the same order: integers such as `[12, -34, 124]` for INT formats, and the real values such as `1.5` for FP formats. So you can read the weights, the multipliers, and the shifts without decoding `model.bin`.

Each layer also has `"ops"`: the integer steps inside it (here, the sums, the bias, the activation, and the requantization). Your device code can follow them, or just run the steps it knows for that class.

## What model.bin holds

All the number arrays sit back to back, each at a multiple of `alignment` bytes. They are packed by these rules:

| Values | How they are stored |
|---|---|
| INT codes | two's complement in their width |
| FP codes | their bit patterns: fp4 as E2M1, ufp4 as E2M2, block scales as E4M3 |
| 4-bit codes | two in each byte, the first one in the low 4 bits |
| 2-bit codes | four in each byte, from the low bits up |
| Other widths | widened: int6 in 8 bits, uint10 in 16 bits |
| 2 bytes or more | little-endian: the bias in 32 bits, the multipliers in 16 bits |

TinyNet's `model.bin` is about 18 KB.

## Choose the layout

By default, conv weights are stored with the channels last: (out, height, width, in). To store some layers with the channels first, give their names or class names:

```python
model_export(model, (image,), "out", layout={"classifier": "channels_first"})
model_export(model, (image,), "out", layout={"QuantDepthwiseConv2d": "channels_first"})
```

A layer inside a block takes the layout of its block. With channels last, a linear layer after a flatten of an image gets its weight columns in (height, width, channel) order, to match the order of the values on your device. In TinyNet, the pool leaves a 1×1 map, so the order does not change.

## Read the files back

To check the files, or to make tables for your device code, read them back in Python:

```python
import json
import numpy as np

info = json.load(open("tiny_net_export/model.json", encoding="utf-8"))
blob = open("tiny_net_export/model.bin", "rb").read()

conv1      = next(layer for layer in info["layers"] if layer.get("name") == "conv1")
weight     = next(item for item in conv1["data"] if item["role"] == "weight")
multiplier = next(item for item in conv1["data"] if item["role"] == "multiplier")

codes = np.frombuffer(blob, dtype=np.int8, count=weight["size"], offset=weight["offset"]).reshape(weight["stored_shape"])
m0    = np.frombuffer(blob, dtype="<u2", count=multiplier["size"] // 2, offset=multiplier["offset"])
print(weight["axes"], codes.shape)  # ['out', 'h', 'w', 'in'] (32, 3, 3, 3)
```

On your device, the same offsets are addresses: if `model.bin` starts at address A in memory, the weights of `conv1` start at A + `offset`. [Device math](../api/device-math.md) lists the integer steps that each layer runs with these numbers.
