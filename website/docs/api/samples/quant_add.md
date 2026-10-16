---
title: "quant_add.py"
description: "QuantAdd: the residual add of two quantized tensors, rounded once."
sidebar_position: 8
---

The quantized residual add. [Residual add](../../learn/residual-add.md) explains the math, and [Shared scales](../../learn/shared-scales.md) the cheaper add with one scale.

```python
from quaver import QuantAdd
```

---

## QuantAdd

```python
QuantAdd(output_format="int8", momentum=0.95, style="mul16_shift", round_mode=None, output_symmetric=True, output_scaler=None)
```

Makes a residual add: both inputs are rescaled to the output scale, added, rounded once, and clipped, as on a device. The output scale follows the range of the sum, or is shared through `output_scaler`. When the two inputs and the output share one `SharedScaler`, the add is a plain integer add with saturation.

| Argument | Type | Meaning |
|---|---|---|
| `output_format` | `str` | The format of the output |
| `momentum` | `float` | The momentum of the range of the sum |
| `style` | `str` | How the device requantizes |
| `round_mode` | `RoundMode \| None` | The rounding rule, or `None` for the usual one |
| `output_symmetric` | `bool` | `False` gives the output a zero point (INT formats only) |
| `output_scaler` | `Scaler \| None` | A scaler for the output, such as a `SharedScaler`; `None` makes a `RangeScaler` |

```python
add = QuantAdd("int8")
```

### forward

```python
add(a, a_params, b, b_params) -> tuple[torch.Tensor, QuantParams]
```

Adds `a` and `b` in the current mode, and returns the sum and its parameters.

| Argument | Type | Meaning |
|---|---|---|
| `a` | `torch.Tensor` | The first input (values or codes) |
| `a_params` | `QuantParams` | Its parameters |
| `b` | `torch.Tensor` | The second input |
| `b_params` | `QuantParams` | Its parameters |

```python
y, y_params = add(x, x_params, residual, residual_params)
```
