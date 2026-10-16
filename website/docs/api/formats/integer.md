---
title: "integer.py"
description: "INT formats of any width: make them, and turn values into codes and back."
sidebar_position: 4
---

INT formats, such as int8 and uint4. `build_format()` makes these by name, so you seldom call this file yourself.

```python
from quaver.formats import integer_decode, integer_encode, integer_format
```

---

## integer_format

```python
integer_format(bits: int, signed: bool) -> NumberFormat
```

Makes an INT format with this many bits. A signed format goes from $-2^{b-1}$ to $2^{b-1} - 1$, and an unsigned one from 0 to $2^b - 1$. Its step is 1.

| Argument | Type | Meaning |
|---|---|---|
| `bits` | `int` | The number of bits |
| `signed` | `bool` | `True` for intN, `False` for uintN |

```python
int4 = integer_format(4, signed=True)
print(int4.name, int4.min_value, int4.max_value)  # int4 -8.0 7.0
```

---

## integer_encode

```python
integer_encode(values: torch.Tensor, number_format: NumberFormat) -> torch.Tensor
```

Turns values on the INT grid into int64 codes. Give it values that are already on the grid, for example from `format_quantize()`.

| Argument | Type | Meaning |
|---|---|---|
| `values` | `torch.Tensor` | Values on the INT grid |
| `number_format` | `NumberFormat` | An INT format |

```python
print(integer_encode(torch.tensor([-3.0, 7.0]), int4))  # tensor([-3,  7])
```

---

## integer_decode

```python
integer_decode(codes: torch.Tensor, number_format: NumberFormat) -> torch.Tensor
```

Turns INT codes back into float32 values.

| Argument | Type | Meaning |
|---|---|---|
| `codes` | `torch.Tensor` | INT codes |
| `number_format` | `NumberFormat` | An INT format |

```python
print(integer_decode(torch.tensor([-3, 7]), int4))  # tensor([-3.,  7.])
```
