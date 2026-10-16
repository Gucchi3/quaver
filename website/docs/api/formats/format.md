---
title: "format.py"
description: "The number formats by name, and the rounding onto them that INT and FP formats share."
sidebar_position: 2
---

The number formats by name, and the functions that round values onto a format and turn them into codes. INT and FP formats use the same code here: only the grid step differs.

```python
from quaver.formats import FORMAT_REGISTRY, build_format, format_codes, format_decode, format_encode, format_quantize, format_round_mode, format_to_integer
```

---

## FORMAT_REGISTRY

```python
FORMAT_REGISTRY: dict[str, NumberFormat]
```

The formats that `build_format()` knows by name. To add a format, make it with `integer_format()` or `fp_format()`, and add it here.

| Name | Format |
|---|---|
| `int2`, `int4`, `int8` | signed integers |
| `uint2`, `uint4`, `uint8` | unsigned integers |
| `uint6`, `uint10` | unsigned integers, for integer block scales |
| `fp4` | FP4 (E2M1): 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6 |
| `ufp4` | unsigned FP4 (E2M2): 0 to 7 in 16 steps |
| `ue4m3` | unsigned E4M3, for the block scales of NVFP4 (up to 448) |

```python
print(list(FORMAT_REGISTRY))
# ['int2', 'int4', 'int8', 'uint2', 'uint4', 'uint6', 'uint8', 'uint10', 'fp4', 'ufp4', 'ue4m3']
```

---

## build_format

```python
build_format(name: str) -> NumberFormat
```

Returns the number format with this name. Besides the names in `FORMAT_REGISTRY`, any INT width works too, such as `"int32"` (the accumulator) or `"uint1"`. An unknown name raises `KeyError` with the list of names.

| Argument | Type | Meaning |
|---|---|---|
| `name` | `str` | A name from `FORMAT_REGISTRY`, or `"intN"` / `"uintN"` for any width N |

```python
from quaver import build_format

int8 = build_format("int8")
print(int8.min_value, int8.max_value)  # -128.0 127.0
print(build_format("int32").name)      # int32
```

---

## format_round_mode

```python
format_round_mode(number_format: NumberFormat, mode: RoundMode | None = None) -> RoundMode
```

Returns `mode`, or the usual rounding of the format when `mode` is `None`: half to even for FP formats (like IEEE floats and NVFP4), and half away from zero for INT formats (like TFLite). Quantizers and layers use it for their default rounding.

| Argument | Type | Meaning |
|---|---|---|
| `number_format` | `NumberFormat` | The format |
| `mode` | `RoundMode \| None` | A rounding rule to use instead, or `None` for the usual one |

```python
print(format_round_mode(build_format("int8")))         # half_away
print(format_round_mode(build_format("fp4")))          # half_even
print(format_round_mode(build_format("fp4"), "floor")) # floor
```

---

## format_quantize

```python
format_quantize(x: torch.Tensor, number_format: NumberFormat, mode: RoundMode, zero_point: torch.Tensor | None = None) -> torch.Tensor
```

Rounds each value of `x` to the nearest value of the format, adds the zero point, and keeps the result inside the range of the format. Give it values that are already divided by the scale. It returns float values on the grid of the format (for fp4: −6, −4, …, 4, 6). The rounding comes before the zero point, as on a device.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values, already divided by the scale |
| `number_format` | `NumberFormat` | The format to round onto |
| `mode` | `RoundMode` | The rounding rule |
| `zero_point` | `torch.Tensor \| None` | The zero point to add after rounding, or `None` for none |

```python
x = torch.tensor([-7.3, -0.26, 0.74, 1.25, 5.0])
print(format_quantize(x, build_format("fp4"), "half_even"))
# tensor([-6.0000, -0.5000,  0.5000,  1.0000,  4.0000])

print(format_quantize(torch.tensor([-1.2, 0.4, 2.6]), build_format("uint8"), "half_away", torch.tensor(10.0)))
# tensor([ 9., 10., 13.])
```

---

## format_codes

```python
format_codes(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, number_format: str, mode: RoundMode) -> torch.Tensor
```

The device op `quantize`: returns the integer codes (int64) of `x / scale` on the format, with the zero point, inside the range. The codes of an FP format are its values divided by `step_min`, so the fp4 value 1.5 is the code 3. It is registered as `torch.ops.quaver.quantize`, which `Quantizer.encode()` calls.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The real values |
| `scale` | `torch.Tensor` | The scale (it broadcasts against `x`) |
| `zero_point` | `torch.Tensor` | The zero point |
| `number_format` | `str` | The name of the format |
| `mode` | `RoundMode` | The rounding rule |

```python
codes = torch.ops.quaver.quantize(torch.tensor([-0.5, 0.1, 0.75]), torch.tensor(0.25), torch.tensor(0.0), "fp4", "half_even")
print(codes)  # tensor([-4,  1,  6])  (fp4 values -2, 0.5, 3)
```

---

## format_encode

```python
format_encode(x: torch.Tensor, number_format: NumberFormat, mode: RoundMode) -> torch.Tensor
```

Rounds `x` onto the format, and returns the codes for storage, as int64: the integer itself for INT formats, and the bit pattern (sign, exponent, mantissa) for FP formats.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values, already divided by the scale |
| `number_format` | `NumberFormat` | The format |
| `mode` | `RoundMode` | The rounding rule |

```python
codes = format_encode(torch.tensor([-6.0, -1.5, 0.0, 0.5, 3.0]), build_format("fp4"), "half_even")
print(codes)  # tensor([15, 11,  0,  1,  5])  (4-bit patterns: -6 is 1111)
```

---

## format_decode

```python
format_decode(codes: torch.Tensor, number_format: NumberFormat) -> torch.Tensor
```

Turns codes for storage (from `format_encode()`) back into float32 values.

| Argument | Type | Meaning |
|---|---|---|
| `codes` | `torch.Tensor` | The stored codes (INT values, or FP bit patterns) |
| `number_format` | `NumberFormat` | The format |

```python
print(format_decode(torch.tensor([15, 11, 0, 1, 5]), build_format("fp4")))
# tensor([-6.0000, -1.5000,  0.0000,  0.5000,  3.0000])
```

---

## format_to_integer

```python
format_to_integer(values: torch.Tensor, number_format: NumberFormat) -> torch.Tensor
```

Turns values on the grid of the format into the integers that exact integer math uses: each value divided by `step_min`, as int64. INT values stay the same. FP4 values are doubled, since its smallest step is 0.5.

| Argument | Type | Meaning |
|---|---|---|
| `values` | `torch.Tensor` | Values on the grid of the format |
| `number_format` | `NumberFormat` | The format |

```python
print(format_to_integer(torch.tensor([-6.0, -1.5, 0.5, 3.0]), build_format("fp4")))
# tensor([-12,  -3,   1,   6])
```
