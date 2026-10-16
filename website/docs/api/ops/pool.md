---
title: "pool.py"
description: "Average pooling on integer codes: add the codes of each window, divide by the count, and round."
sidebar_position: 11
---

Average pooling on codes. The output keeps the scale and the zero point of the input. [Average pooling](../../learn/average-pooling.md) explains it.

```python
from quaver.ops import pool_avg2d_device, pool_avg2d_fake, pool_avg2d_float, pool_avg2d_integer
```

The examples on this page pool one 2×2 window of INT8 codes with the scale 0.1:

```python
codes  = torch.tensor([[[[1, 2], [3, 5]]]])
params = QuantParams(torch.tensor(0.1), torch.tensor(0.0), build_format("int8"))
```

---

## pool_avg2d_integer

```python
pool_avg2d_integer(q_x: torch.Tensor, x_params: QuantParams, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None, mode: RoundMode) -> torch.Tensor
```

The integer mode. Adds the codes of each window (with the zero point removed), divides by the count, rounds, and adds the zero point back.

| Argument | Type | Meaning |
|---|---|---|
| `q_x` | `torch.Tensor` | The codes, (N, C, H, W) |
| `x_params` | `QuantParams` | Their parameters |
| `kernel_size` | `int \| tuple[int, int]` | The size of the window |
| `stride` | `int \| tuple[int, int] \| None` | The stride, or `None` for the window size |
| `mode` | `RoundMode` | The rounding rule (FP formats always round half to even) |

```python
print(pool_avg2d_integer(codes, params, 2, None, "half_away").flatten())  # tensor([3])   (1+2+3+5) / 4 = 2.75
```

---

## pool_avg2d_fake

```python
pool_avg2d_fake(x: torch.Tensor, x_params: QuantParams, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None, mode: RoundMode) -> torch.Tensor
```

The training mode. The same integer math as `pool_avg2d_integer()`, on fake-quantized values, with straight-through gradients.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | Fake-quantized values |
| `x_params` | `QuantParams` | Their parameters |
| `kernel_size` | `int \| tuple[int, int]` | The size of the window |
| `stride` | `int \| tuple[int, int] \| None` | The stride, or `None` for the window size |
| `mode` | `RoundMode` | The rounding rule |

```python
print(pool_avg2d_fake(codes.float() * 0.1, params, 2, None, "half_away").flatten())  # tensor([0.3000])
```

---

## pool_avg2d_float

```python
pool_avg2d_float(x: torch.Tensor, x_params: QuantParams, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None, mode: RoundMode) -> torch.Tensor
```

The float mode (calibration). The float average, without rounding.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | Float values |
| `x_params` | `QuantParams` | Not used |
| `kernel_size` | `int \| tuple[int, int]` | The size of the window |
| `stride` | `int \| tuple[int, int] \| None` | The stride |
| `mode` | `RoundMode` | Not used |

```python
print(pool_avg2d_float(codes.float() * 0.1, params, 2, None, "half_away").flatten())  # tensor([0.2750])
```

---

## pool_avg2d_device

```python
pool_avg2d_device(q_x: torch.Tensor, zero_point: torch.Tensor, number_format: str, kernel_size: list[int], stride: list[int], mode: RoundMode) -> torch.Tensor
```

The device op `pool_avg2d` (`torch.ops.quaver.pool_avg2d`): adds the codes of each window (zero point removed), divides by the count, rounds onto the format, and adds the zero point back.

| Argument | Type | Meaning |
|---|---|---|
| `q_x` | `torch.Tensor` | The codes |
| `zero_point` | `torch.Tensor` | Their zero point |
| `number_format` | `str` | The name of their format |
| `kernel_size` | `list[int]` | The size of the window |
| `stride` | `list[int]` | The stride |
| `mode` | `RoundMode` | The rounding rule |

```python
print(torch.ops.quaver.pool_avg2d(codes, torch.tensor(0), "int8", [2, 2], [2, 2], "half_away").flatten())  # tensor([3])
```
