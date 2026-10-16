---
title: "compute.py"
description: "Turn ranges of values into scales and zero points, find the best range by MSE, and make power-of-two and block scales."
sidebar_position: 2
---

Turns measured ranges into scales. The scalers call these functions.

```python
from quaver.scales import SCALE_MIN, compute_block, compute_mse_range, compute_pow2, compute_scale
```

---

## SCALE_MIN

```python
SCALE_MIN: float
```

The smallest scale (the smallest normal float32, about $1.2 \times 10^{-38}$). A tensor of zeros gets this scale, so no scale is ever 0.

```python
print(SCALE_MIN)  # 1.1754943508222875e-38
```

---

## compute_scale

```python
compute_scale(low: torch.Tensor, high: torch.Tensor, number_format: NumberFormat, symmetric: bool) -> tuple[torch.Tensor, torch.Tensor]
```

Turns a range of values, from `low` to `high`, into a scale and a zero point. Symmetric works for every format. Asymmetric is for INT formats; it widens the range to include 0, so 0 stays exact. [Symmetric](../../learn/symmetric.md) and [asymmetric quantization](../../learn/asymmetric.md) explain the formulas.

| Argument | Type | Meaning |
|---|---|---|
| `low` | `torch.Tensor` | The low end of each range |
| `high` | `torch.Tensor` | The high end of each range |
| `number_format` | `NumberFormat` | The format |
| `symmetric` | `bool` | `True` for a zero point of 0 |

```python
int8 = build_format("int8")
print(compute_scale(torch.tensor(-1.0), torch.tensor(2.0), int8, True))   # (tensor(0.0157), tensor(0.))   2 / 127
print(compute_scale(torch.tensor(-1.0), torch.tensor(2.0), int8, False))  # (tensor(0.0118), tensor(-43.)) 3 / 255
```

---

## compute_mse_range

```python
compute_mse_range(values: torch.Tensor, low: torch.Tensor, high: torch.Tensor, number_format: NumberFormat, symmetric: bool) -> tuple[torch.Tensor, torch.Tensor]
```

Shrinks the range of each group to the one whose quantized values have the smallest squared error. It tries 100 fractions of the measured ends. `LsqScaler` uses it for its start values (`init="mse"`). It makes no GPU sync.

| Argument | Type | Meaning |
|---|---|---|
| `values` | `torch.Tensor` | The values, one row for each group |
| `low` | `torch.Tensor` | The measured low end of each group |
| `high` | `torch.Tensor` | The measured high end of each group |
| `number_format` | `NumberFormat` | The format |
| `symmetric` | `bool` | `True` shrinks both ends together; `False` shrinks each end in turn |

```python
torch.manual_seed(0)
values = torch.cat([torch.randn(1, 1000), torch.tensor([[20.0]])], dim=1)   # one outlier at 20
print(compute_mse_range(values, torch.tensor([-20.0]), torch.tensor([20.0]), build_format("int4"), True))
# (tensor([-8.2000]), tensor([8.2000]))  the outlier is clipped
```

---

## compute_pow2

```python
compute_pow2(scale: torch.Tensor) -> torch.Tensor
```

Rounds each scale up to a power of two, for a device that can only shift. It is exact.

| Argument | Type | Meaning |
|---|---|---|
| `scale` | `torch.Tensor` | The scales |

```python
print(compute_pow2(torch.tensor([0.3, 0.5, 0.0157])))  # tensor([0.5000, 0.5000, 0.0312])
```

---

## compute_block

```python
compute_block(block_amax: torch.Tensor, tensor_scale: torch.Tensor, number_format: NumberFormat, scale_format: NumberFormat) -> torch.Tensor
```

Returns the scale of each block of a block-scaled format: the largest absolute value of the block ÷ the top of the format ÷ the tensor scale, rounded to `scale_format` (half to even). A block scale never goes below the smallest step of `scale_format`, so a block of zeros still has a scale. [Block scales](../../learn/block-scales.md) explains NVFP4.

| Argument | Type | Meaning |
|---|---|---|
| `block_amax` | `torch.Tensor` | The largest absolute value of each block |
| `tensor_scale` | `torch.Tensor` | The tensor scale |
| `number_format` | `NumberFormat` | The format of the elements (its top is 6 for fp4) |
| `scale_format` | `NumberFormat` | The format of the block scales: ue4m3 for NVFP4, or uint6 / uint10 |

```python
print(compute_block(torch.tensor([6.0, 0.9, 100.0]), torch.tensor(0.01), build_format("fp4"), build_format("ue4m3")))
# tensor([ 96.,  15., 448.])  (100 rounds to 96 on the E4M3 grid, and 1667 saturates at 448)
```
