---
title: "dynamic_scaler.py"
description: "DynamicScaler: measure the range of the tensor every time (dynamic quantization)."
sidebar_position: 8
---

A scaler that measures the range of each tensor every time it is called. The sample layers use it for their weights. [Static and dynamic scales](../../learn/static-dynamic.md) explains when to use it.

```python
from quaver import DynamicScaler
from quaver.scales import dynamic_scaler_device
```

---

## DynamicScaler

```python
DynamicScaler(percentile=None, symmetric=True)
```

Makes a scaler that measures the current tensor every time, by its maximum or a percentile. It keeps no state, so it also measures during inference. Give it to a `Quantizer`.

| Argument | Type | Meaning |
|---|---|---|
| `percentile` | `float \| None` | A percentile such as 99.9 to leave out rare large values, or `None` for the maximum |
| `symmetric` | `bool` | `True` for a zero point of 0; `False` for a zero point (INT formats only) |

```python
quantizer = Quantizer(build_format("int8"), DynamicScaler(), Granularity("channel", axis=0))
```

### forward

```python
scaler(x, number_format, granularity, update=True) -> tuple[torch.Tensor, torch.Tensor]
```

Returns the scale and the zero point of `x`, measured now. Both broadcast against `x`. The scale gets no gradient. `Quantizer` calls this for you.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values to measure |
| `number_format` | `NumberFormat` | The format |
| `granularity` | `Granularity` | How many scales |
| `update` | `bool` | Not used: this scaler keeps nothing |

```python
scale, zero_point = DynamicScaler()(torch.tensor([[-1.0, 2.0], [0.5, 4.0]]), build_format("int8"), Granularity("channel", axis=0))
print(scale)  # tensor([[0.0157], [0.0315]])  (2 / 127 and 4 / 127)
```

---

## dynamic_scaler_device

```python
dynamic_scaler_device(x: torch.Tensor, number_format: str, percentile: float | None, symmetric: bool, kind: GranularityKind, axis: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]
```

The op `dynamic_scale` (`torch.ops.quaver.dynamic_scale`): the scale and the zero point of `x` from its range now, for each group. `DynamicScaler` calls it, so an exported model keeps the measurement as one node, with its settings. The math is float, so a device must repeat the same steps.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values to measure |
| `number_format` | `str` | The name of the format |
| `percentile` | `float \| None` | A percentile, or `None` for the maximum |
| `symmetric` | `bool` | `True` for a zero point of 0 |
| `kind` | `GranularityKind` | `"tensor"`, `"channel"`, or `"block"` |
| `axis` | `int` | The axis of the channels or the blocks |
| `block_size` | `int` | The size of a block |

```python
scale, zero_point = torch.ops.quaver.dynamic_scale(torch.tensor([-1.0, 2.0]), "int8", None, True, "tensor", 0, 16)
```
