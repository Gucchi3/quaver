---
title: "wide.py"
description: "Exact rounding of wide products (up to about 80 bits) with int64 math only, for wide accumulators and block scales."
sidebar_position: 5
---

Exact rounding of a × multiplier ÷ (divisor × $2^{\text{shift}}$) with int64 math only, even when the product needs more than 64 bits. The requantization of wide accumulators and of block scales uses it, where float64 would not be exact.

```python
from quaver.ops import wide_round, wide_round_format
```

---

## wide_round

```python
wide_round(a: torch.Tensor, multiplier: torch.Tensor, divisor: torch.Tensor, shift: torch.Tensor, mode: RoundMode) -> torch.Tensor
```

Returns round(a × multiplier ÷ (divisor × $2^{\text{shift}}$)) exactly, with int64 math only. The product can need up to 82 bits, so it is kept in two parts. The tensors broadcast. A result of $2^{40}$ or more is not exact; it only stays that large, so clamp it afterward.

| Argument | Type | Meaning |
|---|---|---|
| `a` | `torch.Tensor` | int64 values, with \|a\| below $2^{62}$ |
| `multiplier` | `torch.Tensor` | int64, from 0 to below $2^{20}$ |
| `divisor` | `torch.Tensor` | int64, from 1 to below $2^{20}$ |
| `shift` | `torch.Tensor` | int64, from −30 up (above 94 counts as 94) |
| `mode` | `RoundMode` | The rounding rule |

```python
print(wide_round(torch.tensor([10**15]), torch.tensor(3), torch.tensor(7), torch.tensor(20), "half_away"))
# tensor([408717564])   10^15 × 3 / (7 × 2^20) = 408717564.2
```

---

## wide_round_format

```python
wide_round_format(a: torch.Tensor, multiplier: torch.Tensor, divisor: torch.Tensor, shift: torch.Tensor, number_format: NumberFormat, mode: RoundMode, zero_point: torch.Tensor | None = None) -> torch.Tensor
```

Rounds a × multiplier ÷ (divisor × $2^{\text{shift}}$), a value in code units, to a code of `number_format` exactly, adds the zero point, and keeps it in the range. INT formats round with `mode`; FP formats round half to even on their grid. It has the same limits as `wide_round()`.

| Argument | Type | Meaning |
|---|---|---|
| `a` | `torch.Tensor` | int64 values |
| `multiplier` | `torch.Tensor` | The multiplier |
| `divisor` | `torch.Tensor` | The divisor |
| `shift` | `torch.Tensor` | The shift |
| `number_format` | `NumberFormat` | The format of the output codes |
| `mode` | `RoundMode` | The rounding rule for INT formats |
| `zero_point` | `torch.Tensor \| None` | The zero point to add, or `None` |

```python
print(wide_round_format(torch.tensor([10**15, -5]), torch.tensor(3), torch.tensor(7), torch.tensor(40), build_format("int8"), "half_away"))
# tensor([127,   0])   the first is clipped to the top of int8
```
