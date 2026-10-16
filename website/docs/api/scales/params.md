---
title: "params.py"
description: "Move between real values and integer codes, the integers of exact sums, and the float type of the training-mode math."
sidebar_position: 6
---

Functions on a tensor and its `QuantParams`: get the codes back from fake-quantized values, turn codes into real values, find the integers that exact sums use, and round in code units. The ops of Quaver are built on these.

```python
from quaver import params_codes, params_values
from quaver.scales import MATH_DTYPES, params_attach, params_bits, params_integers, params_math_dtype, params_named, params_note, params_note_device, params_round, params_unit
```

The examples on this page use:

```python
params8 = QuantParams(torch.tensor(0.5), torch.tensor(3.0), build_format("uint8"))   # scale 0.5, zero point 3
params4 = QuantParams(torch.tensor(0.25), torch.tensor(0.0), build_format("fp4"))
```

---

## params_codes

```python
params_codes(x: torch.Tensor, params: QuantParams, dtype: torch.dtype = torch.float64) -> torch.Tensor
```

Gets the integer codes (int64) back from fake-quantized values. In float64, this is exact as long as the float error of `x` is less than half a step (so turn off TF32).

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | Fake-quantized values |
| `params` | `QuantParams` | Their parameters |
| `dtype` | `torch.dtype` | The float type of the math |

```python
print(params_codes(torch.tensor([-1.5, 0.0, 2.0]), params8))  # tensor([0, 3, 7])
```

---

## params_values

```python
params_values(codes: torch.Tensor, params: QuantParams, dtype: torch.dtype = torch.float64) -> torch.Tensor
```

Turns integer codes back into real values (float32): (code − zero point) × `code_scale`. Use it to read the outputs of the integer mode.

| Argument | Type | Meaning |
|---|---|---|
| `codes` | `torch.Tensor` | Integer codes |
| `params` | `QuantParams` | Their parameters |
| `dtype` | `torch.dtype` | The float type of the math |

```python
print(params_values(torch.tensor([0, 3, 7]), params8))  # tensor([-1.5000,  0.0000,  2.0000])
```

---

## MATH_DTYPES

```python
MATH_DTYPES: dict[str, torch.dtype]
```

The float types of the training-mode math: `"train"` while gradients are on, and `"test"` while they are off. Both are float64 at the start. Change them with `precision_set_math()`, not by hand.

```python
print(MATH_DTYPES)  # {'train': torch.float64, 'test': torch.float64}
```

---

## params_math_dtype

```python
params_math_dtype() -> torch.dtype
```

Returns the float type of the training-mode math right now: the `"train"` type while gradients are on, and the `"test"` type inside `torch.no_grad()`. It only reads a flag on the CPU, so it makes no GPU sync.

```python
print(params_math_dtype())  # torch.float64
```

---

## params_unit

```python
params_unit(params: QuantParams) -> torch.Tensor
```

Returns the real value of one step of the integers that exact sums use. For most tensors this is `code_scale`. For block-scaled tensors, it is the smallest step of the elements × the step of the block scales × the tensor scale, so that all the blocks of a tensor share one unit.

| Argument | Type | Meaning |
|---|---|---|
| `params` | `QuantParams` | The parameters of a tensor |

```python
print(params_unit(params4))  # tensor(0.1250)
```

---

## params_integers

```python
params_integers(codes: torch.Tensor, params: QuantParams) -> torch.Tensor
```

Returns the integers that exact sums use (int64): code − zero point, and for block-scaled tensors, also times the code of the block scale. Real value = integer × `params_unit(params)`.

| Argument | Type | Meaning |
|---|---|---|
| `codes` | `torch.Tensor` | Integer codes |
| `params` | `QuantParams` | Their parameters |

```python
print(params_integers(torch.tensor([0, 3, 7]), params8))  # tensor([-3,  0,  4])
```

---

## params_bits

```python
params_bits(params: QuantParams) -> int
```

Returns how many bits the largest integer of `params_integers()` can need, from the formats only (no data). Exact sums use it to see how wide they get before they run.

| Argument | Type | Meaning |
|---|---|---|
| `params` | `QuantParams` | The parameters of a tensor |

```python
print(params_bits(params8), params_bits(params4))  # 8 5  (fp4 codes go up to 12)
```

---

## params_named

```python
params_named(zero_point: torch.Tensor, number_format: str, block_codes: torch.Tensor | None = None, scale_format: str = "") -> QuantParams
```

Makes `QuantParams` from a zero point and a format name (and block codes), with scales of 1. The device ops get plain tensors and names, and use this to call the integer helpers. Use the result only where the scales do not matter: exact sums and rounding in code units.

| Argument | Type | Meaning |
|---|---|---|
| `zero_point` | `torch.Tensor` | The zero point |
| `number_format` | `str` | The name of the format |
| `block_codes` | `torch.Tensor \| None` | The codes of the block scales, for block-scaled tensors |
| `scale_format` | `str` | The name of the format of the block scales |

```python
params = params_named(torch.tensor(3), "uint8")
print(params.scale, params.zero_point)  # tensor(1.) tensor(3)
```

---

## params_note

```python
params_note(codes: torch.Tensor, params: QuantParams) -> torch.Tensor
```

While `model_export()` captures a model, records the format, the scale, and the zero point of `codes` in the graph, and gives the codes back. At other times, it just returns the codes. The integer ops call it on their outputs, so the export knows the format of every value.

| Argument | Type | Meaning |
|---|---|---|
| `codes` | `torch.Tensor` | The codes of an output |
| `params` | `QuantParams` | Their parameters |

```python
codes = params_note(codes, params)  # the same codes; noted only during the export
```

---

## params_note_device

```python
params_note_device(codes: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, number_format: str, scale_name: str) -> torch.Tensor
```

The op `note` (`torch.ops.quaver.note`): returns a copy of the codes, with their format, scale, and zero point as arguments. `params_note()` calls it only during the export, and the export removes it again, so it costs nothing on the device.

| Argument | Type | Meaning |
|---|---|---|
| `codes` | `torch.Tensor` | The codes |
| `scale` | `torch.Tensor` | Their scale |
| `zero_point` | `torch.Tensor` | Their zero point |
| `number_format` | `str` | The name of their format |
| `scale_name` | `str` | What the scale stands for (`"scale"`, a tensor scale, or the p of NEQ) |

```python
print(torch.ops.quaver.note)  # quaver.note
```

---

## params_round

```python
params_round(value: torch.Tensor, params: QuantParams, mode: RoundMode) -> tuple[torch.Tensor, torch.Tensor]
```

Rounds exact values in code units (before the zero point) to codes of the format, and returns the codes and which values were inside the range. Every integer op ends here, so INT and FP outputs share one path. `mode` is for INT formats; FP formats always round half to even, like FP hardware.

| Argument | Type | Meaning |
|---|---|---|
| `value` | `torch.Tensor` | Exact values in code units |
| `params` | `QuantParams` | The parameters of the output |
| `mode` | `RoundMode` | The rounding rule for INT formats |

```python
codes, inside = params_round(torch.tensor([-200.0, 2.5, 3.5, 300.0]), QuantParams(torch.tensor(1.0), torch.tensor(0.0), build_format("int8")), "half_away")
print(codes, inside)  # tensor([-128,    3,    4,  127]) tensor([False,  True,  True, False])
```

---

## params_attach

```python
params_attach(reference: torch.Tensor, codes: torch.Tensor, params: QuantParams, inside: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor
```

Returns the real values of `codes` in the forward pass, exactly, and the gradients of fake-quantizing `reference` in the backward pass: a straight-through gradient inside the range, and the LSQ gradients for a learned scale and zero point. The training-mode ops call it after their integer math.

| Argument | Type | Meaning |
|---|---|---|
| `reference` | `torch.Tensor` | The float result whose gradient is used |
| `codes` | `torch.Tensor` | The exact codes |
| `params` | `QuantParams` | Their parameters (a learned scale should carry its LSQ gradient scale, as from `Quantizer.measure()`) |
| `inside` | `torch.Tensor` | Where the gradient passes |
| `dtype` | `torch.dtype` | The float type of the math |

```python
reference = torch.tensor([0.7, 1.3], requires_grad=True)
values    = params_attach(reference, torch.tensor([1, 3]), QuantParams(torch.tensor(0.5), torch.tensor(0.0), build_format("int8")), torch.tensor([True, True]))
values.sum().backward()
print(values, reference.grad)  # tensor([0.5000, 1.5000]) tensor([1., 1.])
```
