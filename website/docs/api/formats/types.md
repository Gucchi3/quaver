---
title: "types.py"
description: "NumberFormat, which holds the details of one number format, and RoundMode, the names of the rounding rules."
sidebar_position: 7
---

The types of the number formats: `NumberFormat` describes one format, and `RoundMode` names a rounding rule.

```python
from quaver.formats import NumberFormat, RoundMode
```

---

## NumberFormat

```python
NumberFormat(name, kind, bits, signed, exponent_bits, mantissa_bits, exponent_bias, nan_top, min_value, max_value, step_min)
```

Holds the details of one number format (INT or FP). All the values are on the grid of the format, after a real value is divided by its scale. It is a frozen dataclass, so it cannot change after it is made. You usually get one from `build_format()`, not by calling this yourself.

| Argument | Type | Meaning |
|---|---|---|
| `name` | `str` | The name, such as `"int8"` or `"fp4"` |
| `kind` | `str` | `"int"` or `"fp"` |
| `bits` | `int` | The number of bits of one code |
| `signed` | `bool` | Whether the format has negative values |
| `exponent_bits` | `int` | The exponent bits of an FP format (0 for INT) |
| `mantissa_bits` | `int` | The mantissa bits of an FP format (0 for INT) |
| `exponent_bias` | `int` | The exponent bias of an FP format (0 for INT) |
| `nan_top` | `bool` | Whether the top code stands for NaN (like E4M3), so the largest value is one step lower |
| `min_value` | `float` | The smallest value of the format |
| `max_value` | `float` | The largest value of the format |
| `step_min` | `float` | The smallest step between two values: 1 for INT, 0.5 for fp4. A code of an FP format is its value divided by this |

```python
from quaver import build_format

fp4 = build_format("fp4")
print(fp4.min_value, fp4.max_value, fp4.step_min)  # -6.0 6.0 0.5
```

---

## RoundMode

```python
RoundMode = Literal["half_away", "half_even", "half_up", "floor"]
```

The names of the rounding rules. [round.py](round.md) has a function for each one, and [Rounding](../../learn/rounding.md) explains them.

| Name | Meaning |
|---|---|
| `"half_away"` | To the nearest integer; a half goes away from zero (2.5 → 3, −2.5 → −3) |
| `"half_even"` | To the nearest integer; a half goes to the even one (2.5 → 2, 3.5 → 4) |
| `"half_up"` | To the nearest integer; a half goes up (2.5 → 3, −2.5 → −2) |
| `"floor"` | Down, like a right shift (2.5 → 2, −2.5 → −3) |

```python
quantizer = Quantizer(build_format("int8"), RangeScaler(), round_mode="half_even")
```
