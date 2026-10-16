---
title: "mode.py"
description: "Pick the training, integer, or float version of every op, so a layer writes its forward pass once."
sidebar_position: 12
---

Every op has three versions: one for the training mode (`*_fake`), one for the integer mode (`*_integer`), and one for the float mode (`*_float`). This file groups them into three `OpSet`s, and picks one for the current mode. [Inside a quantized layer](../../guides/inside-a-layer.md) shows a layer that uses it.

```python
from quaver.ops import FAKE_OPS, FLOAT_OPS, INTEGER_OPS, mode_ops
```

---

## FAKE_OPS, INTEGER_OPS, FLOAT_OPS

```python
FAKE_OPS: OpSet
INTEGER_OPS: OpSet
FLOAT_OPS: OpSet
```

The three sets of ops: the training versions, the integer versions, and the float versions.

| Set | Its ops | Mode |
|---|---|---|
| `FAKE_OPS` | `requantize_fake`, `add_fake`, … | training mode: fake-quantized values, the same integer math, float gradients |
| `INTEGER_OPS` | `requantize_integer`, `add_integer`, … | integer mode: integer codes, integer math only |
| `FLOAT_OPS` | `requantize_float`, `add_float`, … | float mode: float values, no rounding (calibration) |

```python
print(INTEGER_OPS.requantize.__name__)  # requantize_integer
```

---

## mode_ops

```python
mode_ops(integer_mode: bool, float_mode: bool = False) -> OpSet
```

Returns the set of ops for the mode: `INTEGER_OPS` when `integer_mode` is true, `FLOAT_OPS` when `float_mode` is true, and `FAKE_OPS` otherwise. `integer_mode` wins over `float_mode`. Call it at the start of `forward()`, with the two flags of your layer.

| Argument | Type | Meaning |
|---|---|---|
| `integer_mode` | `bool` | The `integer_mode` flag of the layer |
| `float_mode` | `bool` | The `float_mode` flag of the layer |

```python
def forward(self, x, x_params):
    ops = mode_ops(self.integer_mode, self.float_mode)
    ...
    return ops.requantize(acc, acc_params, out_params, self.style), out_params
```
