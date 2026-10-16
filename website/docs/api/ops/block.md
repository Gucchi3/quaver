---
title: "block.py"
description: "Quantize codes to a block-scaled format (NVFP4 and the like) with integer math, as a device does while it runs."
sidebar_position: 6
---

Turns codes with one fixed scale (such as an accumulator) into a block-scaled format, such as NVFP4, as a device does at run time: each block gets its scale from its largest code, and each element is divided by its block scale. Everything is integer math. [Block scales](../../learn/block-scales.md) explains the steps.

```python
from quaver.ops import block_quantize_device, block_quantize_fake, block_quantize_float, block_quantize_integer
```

The examples on this page turn INT8 codes (scale 0.02) into NVFP4, with a tensor scale of 0.002:

```python
torch.manual_seed(0)
codes       = torch.randint(-127, 128, (1, 32))
params      = QuantParams(torch.tensor(0.02), torch.tensor(0.0), build_format("int8"))
fp4, ue4m3  = build_format("fp4"), build_format("ue4m3")
blocks      = Granularity("block", axis=-1, block_size=16)
```

---

## block_quantize_integer

```python
block_quantize_integer(q: torch.Tensor, q_params: QuantParams, number_format: NumberFormat, scale_format: NumberFormat, tensor_scale: torch.Tensor, granularity: Granularity, style_name: str) -> tuple[torch.Tensor, QuantParams]
```

The integer mode. Quantizes codes with one fixed scale to a block-scaled format. The code of each block scale = |largest code of the block| × $M_6 / 2^{n_6}$, rounded half to even onto `scale_format`. Each element = code × $M_r$ / (block scale code × $2^{n_r}$), rounded half to even. Returns the element codes and their `QuantParams` (with the block scales in `.block`). Exact for any width of `q`.

| Argument | Type | Meaning |
|---|---|---|
| `q` | `torch.Tensor` | The codes (of a format, or an accumulator of any width) |
| `q_params` | `QuantParams` | Their parameters (one scale for the tensor) |
| `number_format` | `NumberFormat` | The format of the elements, such as fp4 |
| `scale_format` | `NumberFormat` | The format of the block scales, such as ue4m3 |
| `tensor_scale` | `torch.Tensor` | The fixed tensor scale |
| `granularity` | `Granularity` | The blocks |
| `style_name` | `str` | The requantization style, for the multipliers |

```python
fp4_codes, fp4_params = block_quantize_integer(codes, params, fp4, ue4m3, torch.tensor(0.002), blocks, "mul16_shift")
print(fp4_codes[0, :8], fp4_params.number_format.name)  # tensor([ 6, -8, 12,  1,  8,  8, -6,  4]) fp4   (fp4 values × 2)
```

---

## block_quantize_fake

```python
block_quantize_fake(x: torch.Tensor, x_params: QuantParams, number_format: NumberFormat, scale_format: NumberFormat, tensor_scale: torch.Tensor, granularity: Granularity, style_name: str) -> tuple[torch.Tensor, QuantParams]
```

The training mode. The same integer math as `block_quantize_integer()`, on fake-quantized values, with straight-through gradients where the value fits under the top of its block. `x` must hold exact values, so turn off TF32.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | Fake-quantized values with one scale |
| `x_params` | `QuantParams` | Their parameters |
| `number_format` | `NumberFormat` | The format of the elements |
| `scale_format` | `NumberFormat` | The format of the block scales |
| `tensor_scale` | `torch.Tensor` | The fixed tensor scale |
| `granularity` | `Granularity` | The blocks |
| `style_name` | `str` | The requantization style |

```python
values, value_params = block_quantize_fake(codes.float() * 0.02, params, fp4, ue4m3, torch.tensor(0.002), blocks, "mul16_shift")
# values are the real values of the same fp4 codes as above
```

---

## block_quantize_float

```python
block_quantize_float(x: torch.Tensor, x_params: QuantParams, number_format: NumberFormat, scale_format: NumberFormat, tensor_scale: torch.Tensor, granularity: Granularity, style_name: str) -> tuple[torch.Tensor, QuantParams]
```

The float mode (calibration). Returns `x` as it is, with the block scales that its float values would get.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | Float values |
| `x_params` | `QuantParams` | Their parameters |
| `number_format` | `NumberFormat` | The format of the elements |
| `scale_format` | `NumberFormat` | The format of the block scales |
| `tensor_scale` | `torch.Tensor` | The tensor scale |
| `granularity` | `Granularity` | The blocks |
| `style_name` | `str` | Not used |

```python
values, value_params = block_quantize_float(codes.float(), params, fp4, ue4m3, torch.tensor(0.002), blocks, "mul16_shift")
print(value_params.scale.shape)  # torch.Size([1, 32])
```

---

## block_quantize_device

```python
block_quantize_device(q: torch.Tensor, zero_point: torch.Tensor, scale_multiplier: torch.Tensor, scale_shift: torch.Tensor, value_multiplier: torch.Tensor, value_shift: torch.Tensor, out_format: str, scale_format: str, kind: GranularityKind, axis: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]
```

The device op `block_quantize` (`torch.ops.quaver.block_quantize`): the code of each block scale from the largest |code| of its block, then each element divided by its block code, both rounded half to even. Returns the element codes and the block codes (spread so that they broadcast against `q`). `block_quantize_integer()` finds the fixed multipliers and shifts and calls it.

| Argument | Type | Meaning |
|---|---|---|
| `q` | `torch.Tensor` | The codes |
| `zero_point` | `torch.Tensor` | Their zero point |
| `scale_multiplier`, `scale_shift` | `torch.Tensor` | $M_6$ and $n_6$: one step of `q` in steps of the block scales, at the top of `out_format` |
| `value_multiplier`, `value_shift` | `torch.Tensor` | $M_r$ and $n_r$: one step of `q` in steps of the elements |
| `out_format` | `str` | The name of the element format |
| `scale_format` | `str` | The name of the block-scale format |
| `kind` | `GranularityKind` | `"block"` |
| `axis` | `int` | The axis of the blocks |
| `block_size` | `int` | The size of a block |

```python
print(torch.ops.quaver.block_quantize)  # quaver.block_quantize
```
