---
title: "pack.py"
description: "pack_codes(): pack codes into bytes as the device memory holds them."
sidebar_position: 4
---

Packs codes into bytes, by the rules of `model.bin`: INT codes in two's complement, FP codes as their bit patterns, 2-bit and 4-bit codes from the low bits up, other widths widened, and little-endian.

```python
from quaver.export import PACK_WIDTHS, pack_codes
```

---

## PACK_WIDTHS

```python
PACK_WIDTHS: tuple[int, ...]   # (2, 4, 8, 16, 32, 64)
```

The widths in memory. 2-bit and 4-bit codes are packed several to a byte; every other width takes the next of 8, 16, 32, and 64 bits (int6 in 8 bits, uint10 in 16).

```python
print(PACK_WIDTHS)  # (2, 4, 8, 16, 32, 64)
```

---

## pack_codes

```python
pack_codes(codes: torch.Tensor, number_format: str) -> tuple[bytes, dict[str, Any]]
```

Packs codes, in the order of their elements, into bytes, and returns the bytes with how they are stored (the encoding and the width in bits). INT codes are two's complement in their width. FP codes (the grid integers of Quaver, such as 3 for fp4 1.5) are stored as their bit patterns. `"float32"` and `"float64"` store floats as IEEE 754. Values out of the range of the format raise `ValueError`.

| Argument | Type | Meaning |
|---|---|---|
| `codes` | `torch.Tensor` | The codes, in the order to store them |
| `number_format` | `str` | The name of their format, or `"float32"` / `"float64"` |

```python
data, info = pack_codes(torch.tensor([1, -2, 7, -8]), "int4")
print(data.hex(" "), info["bits"])   # e1 87 4     the first code in the low 4 bits: 1 and -2 (0xE) make 0xE1

data, info = pack_codes(torch.tensor([3, -12, 1]), "fp4")   # fp4 values 1.5, -6, 0.5
print(data.hex(" "))                 # f3 01       0011 and 1111 in the first byte, 0001 in the second

data, info = pack_codes(torch.tensor([300, -2]), "int32")
print(data.hex(" "))                 # 2c 01 00 00 fe ff ff ff   little-endian
```
