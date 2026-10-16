---
title: "model_export.py"
description: "model_export(): write model.json and model.bin for the device, and the tables it uses."
sidebar_position: 2
---

The export itself. It captures the model with `torch.export`, computes every value that does not depend on the input (weight codes, bias, multipliers and shifts, zero points, block codes), and writes them with the structure of the model.

```python
from quaver import model_export
from quaver.export import EXPORT_FORMATS, EXPORT_LAYOUTS, EXPORT_OUTPUTS, EXPORT_ROLES, EXPORT_SETTINGS
```

---

## model_export

```python
model_export(model: nn.Module, inputs: tuple[torch.Tensor, ...], folder: str | Path, alignment: int = 4, layout: dict[str, ExportLayout] | None = None, json_values: bool = True) -> dict[str, Any]
```

Exports a model in the integer mode to `folder/model.json` and `folder/model.bin`, and returns the content of `model.json` as a dict. Put the model in `eval()` and in the integer mode first. `forward()` may return tensors, pairs (codes, `QuantParams`), or a tuple or list of them; a pair also records the scale and the zero point of the output.

- `model.json` lists the layers in order (a block holds `"layers"`, other layers their `"ops"`), their inputs and outputs by name with format and shape, their settings, and for every data item its role, format, encoding, shape, layout, offset, and size.
- `model.bin` holds every data item, packed back to back, each starting at a multiple of `alignment` bytes. An offset is the address counted from the start of the file.
- With channels last, when a flatten of the activations feeds a linear layer, the columns of its weight are reordered to (spatial…, channel), so the device result stays the same.

| Argument | Type | Meaning |
|---|---|---|
| `model` | `nn.Module` | The trained model, in `eval()` and in the integer mode |
| `inputs` | `tuple[torch.Tensor, ...]` | Example inputs for `forward()` (only their shape and type matter) |
| `folder` | `str \| Path` | The folder to write the two files into |
| `alignment` | `int` | Each data item starts at a multiple of this many bytes |
| `layout` | `dict[str, ExportLayout] \| None` | Layer paths or class names mapped to `"channels_last"` (default) or `"channels_first"`; a child layer takes the layout of its block |
| `json_values` | `bool` | `True` also writes the numbers of every data item in decimal (`"values"`): INT codes as integers, FP codes as their real values |

```python
model.eval()
switch_integer_mode(model, True)
info = model_export(model, (torch.randn(1, 3, 8, 8),), "net_export", alignment=16, layout={"QuantConv2d": "chw"}, json_values=False)
print(info["size"], [layer.get("class", layer.get("op")) for layer in info["layers"]])
# 385 ['quaver.quantize', 'QuantConv2d']
```

---

## EXPORT_LAYOUTS

```python
EXPORT_LAYOUTS: dict[str, str]
```

The names that `layout` takes, and the layout each one means: `"channels_last"` and its 2-D name `"hwc"`, and `"channels_first"` and its 2-D name `"chw"`.

```python
print(EXPORT_LAYOUTS)
# {'channels_last': 'channels_last', 'channels_first': 'channels_first', 'hwc': 'channels_last', 'chw': 'channels_first'}
```

---

## EXPORT_SETTINGS

```python
EXPORT_SETTINGS: tuple[str, ...]
```

The attribute names of a layer that go to its `"settings"` in `model.json`, when the layer has them. To export a setting of your own layer, store it under one of these names, or add its name here.

```python
print(EXPORT_SETTINGS)
# ('stride', 'padding', 'dilation', 'groups', 'activation', 'style', 'round_mode', 'heads', 'key_dim', 'value_dim', 'neq_scope', 'output_block', 'kernel_size')
```

---

## EXPORT_ROLES

```python
EXPORT_ROLES: dict[str, dict[str, str]]
```

The role name of each constant argument of each device op. A data item is named by its role, the same in every layer of a class (`weight`, `bias`, `multiplier`, `shift`, `output_zero_point`, …). Your own layer gets the same names when it uses the same ops. To add a device op to the export, add its roles here.

```python
print(EXPORT_ROLES["requantize"])
# {'zero_point': 'accumulator_zero_point', 'multiplier': 'multiplier', 'shift': 'shift', 'out_zero_point': 'output_zero_point'}
```

---

## EXPORT_FORMATS

```python
EXPORT_FORMATS: dict[str, dict[str | int, str]]
```

The format of the codes of each device op: for each tensor argument (by name) and each output (by index), the argument that names its format, or the name of the format itself. Multipliers are `uint16`, and shifts are `int8`. To add a device op to the export, add its formats here.

```python
print(EXPORT_FORMATS["requantize"])
# {'multiplier': 'uint16', 'shift': 'int8', 'out_zero_point': 'out_format', 0: 'out_format'}
```

---

## EXPORT_OUTPUTS

```python
EXPORT_OUTPUTS: dict[str, dict[int, str]]
```

The role names of the outputs of a device op after the first one, such as the integer p of NEQ, so that they get readable names in `model.json`.

```python
print(EXPORT_OUTPUTS)  # {'block_quantize': {1: 'block_codes'}, 'normalize_l1': {1: 'peak'}}
```
