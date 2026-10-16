---
title: "add.py"
description: "The residual add of two quantized tensors with different scales, rounded once."
sidebar_position: 8
---

Adds two quantized tensors with different scales: each input is rescaled to the output with its own multiplier and shift, the two are added, and the sum is rounded once. [Residual add](../../learn/residual-add.md) explains the math.

```python
from quaver.ops import add_device, add_fake, add_float, add_integer
```

The examples on this page add `a` (scale 0.1) and `b` (scale 0.05) into an INT8 output (scale 0.2):

```python
int8       = build_format("int8")
a_params   = QuantParams(torch.tensor(0.1) , torch.tensor(0.0), int8)
b_params   = QuantParams(torch.tensor(0.05), torch.tensor(0.0), int8)
out_params = QuantParams(torch.tensor(0.2) , torch.tensor(0.0), int8)
```

---

## add_integer

```python
add_integer(q_a: torch.Tensor, a_params: QuantParams, q_b: torch.Tensor, b_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor
```

The integer mode. Rescales both inputs to the code units of the output without rounding, adds them, and rounds the sum once. On a device: (code − zero point) × $M_0$ of each input is brought to the larger shift, the two are added, and the sum is shifted once with rounding.

| Argument | Type | Meaning |
|---|---|---|
| `q_a` | `torch.Tensor` | The codes of `a` |
| `a_params` | `QuantParams` | Their parameters |
| `q_b` | `torch.Tensor` | The codes of `b` |
| `b_params` | `QuantParams` | Their parameters |
| `out_params` | `QuantParams` | The parameters of the output |
| `style_name` | `str` | A name from `REQUANTIZE_REGISTRY` |

```python
print(add_integer(torch.tensor([10, -30]), a_params, torch.tensor([7, 7]), b_params, out_params, "mul16_shift"))
# tensor([  7, -13])   (1.0 + 0.35) / 0.2 = 6.75, and (-3.0 + 0.35) / 0.2 = -13.25
```

---

## add_fake

```python
add_fake(a: torch.Tensor, a_params: QuantParams, b: torch.Tensor, b_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor
```

The training mode. The same integer math as `add_integer()`, on fake-quantized values, with straight-through gradients. A learned output scale (LSQ) also gets its gradient.

| Argument | Type | Meaning |
|---|---|---|
| `a` | `torch.Tensor` | Fake-quantized values of `a` |
| `a_params` | `QuantParams` | Their parameters |
| `b` | `torch.Tensor` | Fake-quantized values of `b` |
| `b_params` | `QuantParams` | Their parameters |
| `out_params` | `QuantParams` | The parameters of the output |
| `style_name` | `str` | A name from `REQUANTIZE_REGISTRY` |

```python
print(add_fake(torch.tensor([1.0, -3.0]), a_params, torch.tensor([0.35, 0.35]), b_params, out_params, "mul16_shift"))
# tensor([ 1.4000, -2.6000])   the codes 7 and -13 times 0.2
```

---

## add_float

```python
add_float(a: torch.Tensor, a_params: QuantParams, b: torch.Tensor, b_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor
```

The float mode (calibration). The float sum `a + b`, without requantizing.

| Argument | Type | Meaning |
|---|---|---|
| `a`, `b` | `torch.Tensor` | Float values |
| `a_params`, `b_params`, `out_params` | `QuantParams` | Not used |
| `style_name` | `str` | Not used |

```python
print(add_float(torch.tensor([1.0]), a_params, torch.tensor([0.35]), b_params, out_params, "mul16_shift"))  # tensor([1.3500])
```

---

## add_device

```python
add_device(q_a: torch.Tensor, a_zero_point: torch.Tensor, a_multiplier: torch.Tensor, a_shift: torch.Tensor, q_b: torch.Tensor, b_zero_point: torch.Tensor, b_multiplier: torch.Tensor, b_shift: torch.Tensor, out_zero_point: torch.Tensor, out_format: str, style_name: str) -> torch.Tensor
```

The device op `add` (`torch.ops.quaver.add`): (code − zero point) × $M_0 / 2^n$ of each input, added without rounding, then rounded once onto `out_format`, plus the output zero point. `add_integer()` calls it with the fixed $M_0$ and $n$ of each input (input scale ÷ output scale).

| Argument | Type | Meaning |
|---|---|---|
| `q_a`, `q_b` | `torch.Tensor` | The codes of the inputs |
| `a_zero_point`, `b_zero_point` | `torch.Tensor` | Their zero points |
| `a_multiplier`, `b_multiplier` | `torch.Tensor` | Their $M_0$ |
| `a_shift`, `b_shift` | `torch.Tensor` | Their $n$ |
| `out_zero_point` | `torch.Tensor` | The zero point of the output |
| `out_format` | `str` | The name of the output format |
| `style_name` | `str` | The style, for the rounding |

```python
# a: 0.1 / 0.2 = 0.5 = 32768 / 2^16, b: 0.05 / 0.2 = 0.25 = 32768 / 2^17
codes = torch.ops.quaver.add(torch.tensor([10]), torch.tensor(0), torch.tensor(32768), torch.tensor(16), torch.tensor([7]), torch.tensor(0), torch.tensor(32768), torch.tensor(17), torch.tensor(0), "int8", "mul16_shift")
print(codes)  # tensor([7])
```
