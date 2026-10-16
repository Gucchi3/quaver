---
title: "requantize.py"
description: "Rescale accumulators to output codes with an integer multiplier and a shift, as a device does."
sidebar_position: 4
---

Requantization: turn an accumulator (or any codes) into the codes of the output, with an integer multiplier $M_0$ and a shift $n$. [Requantization](../../learn/requantization.md) explains the math.

$$
q_y = \mathrm{clamp}\Big( \big(((acc - Z_{acc}) \cdot M_0 + 2^{n-1}) \gg n\big) + Z_y \Big)
$$

```python
from quaver.ops import REQUANTIZE_REGISTRY, build_requantize, requantize_device, requantize_fake, requantize_float, requantize_integer, requantize_params, requantize_ratio, requantize_rescale
```

The examples on this page use an accumulator with the scale 0.0005 and an INT8 output with the scale 0.25, so the ratio is 0.002:

```python
acc_params = QuantParams(torch.tensor(0.0005), torch.tensor(0.0), build_format("int32"))
out_params = QuantParams(torch.tensor(0.25), torch.tensor(0.0), build_format("int8"))
```

---

## REQUANTIZE_REGISTRY

```python
REQUANTIZE_REGISTRY: dict[str, RequantizeStyle]
```

The ways a device can requantize, by name. Layers take the name as `style`. To match a device that works another way, such as the 31-bit multiplier of CMSIS-NN, add its style here.

| Name | Multiplier | Rounding | Math |
|---|---|---|---|
| `"mul16_shift"` | up to $2^{16} - 1$ | half up | $(acc \cdot M_0 + 2^{n-1}) \gg n$ (the default) |
| `"shift_only"` | 1 | half up | $(acc + 2^{n-1}) \gg n$: scales must be powers of two |
| `"exact"` | none | half away | the real ratio in float64 (not for a device) |

```python
print(list(REQUANTIZE_REGISTRY))  # ['mul16_shift', 'shift_only', 'exact']
```

---

## build_requantize

```python
build_requantize(name: str) -> RequantizeStyle
```

Returns the style with this name. An unknown name raises `KeyError` with the list of names.

| Argument | Type | Meaning |
|---|---|---|
| `name` | `str` | A name from `REQUANTIZE_REGISTRY` |

```python
print(build_requantize("mul16_shift").multiplier_max)  # 65535
```

---

## requantize_params

```python
requantize_params(ratio: torch.Tensor, style_name: str) -> FixedPoint
```

Finds the multiplier $M_0$ and the shift $n$ for a ratio of fixed scales, so that ratio ≈ $M_0 / 2^n$. $n$ is the largest shift that keeps $M_0$ within the multiplier (16 bits for `"mul16_shift"`), so $M_0$ uses every bit. A negative $n$ is a left shift, for ratios of $2^{16}$ or more. Use it only for fixed scales; ratios that change with the data use `requantize_ratio()`.

| Argument | Type | Meaning |
|---|---|---|
| `ratio` | `torch.Tensor` | The ratio of scales, such as input scale × weight scale ÷ output scale (float64) |
| `style_name` | `str` | A name from `REQUANTIZE_REGISTRY` |

```python
fixed = requantize_params(torch.tensor(0.002, dtype=torch.float64), "mul16_shift")
print(fixed.multiplier, fixed.shift)  # tensor(33554) tensor(24)   33554 / 2^24 = 0.0019999...
```

---

## requantize_ratio

```python
requantize_ratio(numerator: torch.Tensor, denominator: torch.Tensor, style_name: str) -> FixedPoint
```

Finds $M_0$ and $n$ for numerator ÷ denominator with integers only (shifts, a multiplication, and an integer division), as a device does while it runs. NEQ and the attention map of SimA use it for ratios that change with the data. $M_0$ = round(numerator × $2^n$ ÷ denominator), with a half rounded up.

| Argument | Type | Meaning |
|---|---|---|
| `numerator` | `torch.Tensor` | An int64 tensor, from 1 to below $2^{62}$ |
| `denominator` | `torch.Tensor` | An int64 tensor, from 1 to below $2^{46}$ |
| `style_name` | `str` | A style with a multiplier (not `"exact"`) |

```python
fixed = requantize_ratio(torch.tensor(3), torch.tensor(1000), "mul16_shift")
print(fixed.multiplier, fixed.shift)  # tensor(50332) tensor(24)   50332 / 2^24 = 0.003
```

---

## requantize_rescale

```python
requantize_rescale(acc: torch.Tensor, fixed_point: FixedPoint, dtype: torch.dtype = torch.float64) -> torch.Tensor
```

Multiplies the accumulator by the ratio, $acc \cdot M_0 / 2^n$, before rounding, in the float type `dtype`. In float64 this is exact for a 32-bit accumulator and a 16-bit multiplier.

| Argument | Type | Meaning |
|---|---|---|
| `acc` | `torch.Tensor` | The accumulator |
| `fixed_point` | `FixedPoint` | The multiplier and the shift |
| `dtype` | `torch.dtype` | The float type of the math |

```python
print(requantize_rescale(torch.tensor([1000, -2500]), fixed))  # tensor([ 2.0000, -4.9999], dtype=torch.float64)
```

---

## requantize_integer

```python
requantize_integer(acc: torch.Tensor, acc_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor
```

The integer mode. Rescales the accumulator (or any codes) to the codes of the output, as a device does: the zero point of the input is removed first, the ratio of the code scales becomes $M_0$ and $n$, and the product is rounded once with int64 math, exact for any width.

| Argument | Type | Meaning |
|---|---|---|
| `acc` | `torch.Tensor` | The accumulator or codes (int64) |
| `acc_params` | `QuantParams` | Their parameters |
| `out_params` | `QuantParams` | The parameters of the output |
| `style_name` | `str` | A name from `REQUANTIZE_REGISTRY` |

```python
print(requantize_integer(torch.tensor([1000, -2500, 70000]), acc_params, out_params, "mul16_shift"))
# tensor([  2,  -5, 127])   70000 × 0.002 = 140 is clipped
```

---

## requantize_fake

```python
requantize_fake(x: torch.Tensor, x_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor
```

The training mode. Gets the codes back from `x`, runs the same integer math as `requantize_integer()`, and returns the real values of the output codes, with a straight-through gradient (and the LSQ gradient for a learned output scale). `x` must hold exact values, so turn off TF32.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The fake-quantized accumulator |
| `x_params` | `QuantParams` | Its parameters |
| `out_params` | `QuantParams` | The parameters of the output |
| `style_name` | `str` | A name from `REQUANTIZE_REGISTRY` |

```python
x = torch.tensor([0.5, -1.25, 35.0], requires_grad=True)   # the accumulator 1000, -2500, 70000 times 0.0005
y = requantize_fake(x, acc_params, out_params, "mul16_shift")
y.sum().backward()
print(y, x.grad)  # tensor([ 0.5000, -1.2500, 31.7500]) tensor([1., 1., 0.])
```

---

## requantize_float

```python
requantize_float(x: torch.Tensor, x_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor
```

The float mode (calibration). Returns `x` as it is, without requantizing, so the scalers see the float values.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The float values |
| `x_params` | `QuantParams` | Not used |
| `out_params` | `QuantParams` | Not used |
| `style_name` | `str` | Not used |

```python
print(requantize_float(torch.tensor([0.5, 35.0]), acc_params, out_params, "mul16_shift"))  # tensor([ 0.5000, 35.0000])
```

---

## requantize_device

```python
requantize_device(acc: torch.Tensor, zero_point: torch.Tensor, multiplier: torch.Tensor, shift: torch.Tensor, out_zero_point: torch.Tensor, out_format: str, mode: RoundMode) -> torch.Tensor
```

The device op `requantize` (`torch.ops.quaver.requantize`): (acc − zero_point) × $M_0 / 2^n$, rounded once onto `out_format`, plus the output zero point, inside the range. These are exactly the numbers a device gets: `requantize_integer()` calls it with the $M_0$ and $n$ it found.

| Argument | Type | Meaning |
|---|---|---|
| `acc` | `torch.Tensor` | The accumulator (int64) |
| `zero_point` | `torch.Tensor` | The zero point of the accumulator |
| `multiplier` | `torch.Tensor` | $M_0$, one or one for each channel |
| `shift` | `torch.Tensor` | $n$, one or one for each channel |
| `out_zero_point` | `torch.Tensor` | The zero point of the output |
| `out_format` | `str` | The name of the output format |
| `mode` | `RoundMode` | The rounding of the shift |

```python
codes = torch.ops.quaver.requantize(torch.tensor([1000, -2500, 70000]), torch.tensor(0), torch.tensor(33554), torch.tensor(24), torch.tensor(0), "int8", "half_up")
print(codes)  # tensor([  2,  -5, 127])
```
