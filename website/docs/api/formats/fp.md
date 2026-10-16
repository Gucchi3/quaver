---
title: "fp.py"
description: "Small FP formats (FP4, UFP4, and UE4M3 for block scales): make them, find their steps, and turn values into bit codes and back."
sidebar_position: 3
---

Small floating-point formats: FP4, UFP4, and UE4M3 for the block scales of NVFP4. [INT and FP formats](../../learn/int-fp.md) explains how they work.

```python
from quaver.formats import fp_decode, fp_encode, fp_format, fp_step, fp_values
```

---

## fp_format

```python
fp_format(exponent_bits: int, mantissa_bits: int, signed: bool, nan_top: bool, name: str = "") -> NumberFormat
```

Makes a small FP format from its exponent bits and mantissa bits. An unsigned format uses the place of the sign bit as one more mantissa bit, so UFP4 is E2M2. It has subnormals and no infinity.

| Argument | Type | Meaning |
|---|---|---|
| `exponent_bits` | `int` | The bits of the exponent |
| `mantissa_bits` | `int` | The bits of the mantissa |
| `signed` | `bool` | Whether the format has a sign bit |
| `nan_top` | `bool` | `True` keeps the top code for NaN (like E4M3), so the largest value is one step lower |
| `name` | `str` | The name. Empty gives `fp` or `ufp` and the number of bits, such as `"fp4"` |

```python
ue4m3 = fp_format(4, 3, signed=False, nan_top=True, name="ue4m3")
print(ue4m3.max_value, ue4m3.step_min)  # 448.0 0.001953125
```

---

## fp_step

```python
fp_step(x: torch.Tensor, number_format: NumberFormat) -> torch.Tensor
```

Returns the step of the FP grid at each value of `x`. Values from $2^e$ up to $2^{e+1}$ have the step $2^{e - m}$, where $m$ is the number of mantissa bits. Very small values use the step of the lowest exponent. It is exact.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values, already divided by the scale |
| `number_format` | `NumberFormat` | An FP format |

```python
print(fp_step(torch.tensor([0.3, 1.2, 2.5, 5.0]), build_format("fp4")))
# tensor([0.5000, 0.5000, 1.0000, 2.0000])
```

---

## fp_encode

```python
fp_encode(values: torch.Tensor, number_format: NumberFormat) -> torch.Tensor
```

Turns values on the FP grid into their bit patterns (sign, exponent, mantissa), as int64. Give it values that are already on the grid, for example from `format_quantize()`.

| Argument | Type | Meaning |
|---|---|---|
| `values` | `torch.Tensor` | Values on the grid of the format |
| `number_format` | `NumberFormat` | An FP format |

```python
print(fp_encode(torch.tensor([1.5, -6.0, 0.5]), build_format("fp4")))
# tensor([ 3, 15,  1])  (0011, 1111, 0001)
```

---

## fp_decode

```python
fp_decode(codes: torch.Tensor, number_format: NumberFormat) -> torch.Tensor
```

Turns FP bit patterns back into float32 values. For a format with `nan_top`, the top code becomes NaN.

| Argument | Type | Meaning |
|---|---|---|
| `codes` | `torch.Tensor` | Bit patterns, as integers |
| `number_format` | `NumberFormat` | An FP format |

```python
print(fp_decode(torch.tensor([3, 15, 1]), build_format("fp4")))
# tensor([ 1.5000, -6.0000,  0.5000])
```

---

## fp_values

```python
fp_values(number_format: NumberFormat) -> torch.Tensor
```

Returns every value that the FP format can hold, sorted, without NaN (−0 and 0 count as one). Useful for tests and lookup tables.

| Argument | Type | Meaning |
|---|---|---|
| `number_format` | `NumberFormat` | An FP format |

```python
print(fp_values(build_format("ufp4")))
# tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.2500, 1.5000, 1.7500, 2.0000,
#         2.5000, 3.0000, 3.5000, 4.0000, 5.0000, 6.0000, 7.0000])
```
