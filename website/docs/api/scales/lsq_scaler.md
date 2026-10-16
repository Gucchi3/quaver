---
title: "lsq_scaler.py"
description: "LsqScaler: learn the scale with the loss (LSQ), and the zero point too (LSQ+)."
sidebar_position: 11
---

A scaler whose scale is a parameter that the optimizer trains. [QAT, STE, and LSQ](../../learn/qat.md) explains LSQ.

```python
from quaver import LsqScaler
from quaver.scales import STEP_MIN
```

---

## STEP_MIN

```python
STEP_MIN: float
```

The smallest learned step (the float32 epsilon, about $1.2 \times 10^{-7}$). `lsq_clamp_steps()` keeps every step above it, since one optimizer step can push a small step below 0.

```python
print(STEP_MIN)  # 1.1920928955078125e-07
```

---

## LsqScaler

```python
LsqScaler(scale_count=1, symmetric=True, init="mse")
```

Makes a scaler that learns its scale (the step) with gradient descent. With `symmetric=False`, it also learns the zero point (LSQ+). The step is NaN until the first training batch, which sets the start values. Use `lsq_param_groups()` for the optimizer, and call `lsq_clamp_steps()` after each optimizer step.

| Argument | Type | Meaning |
|---|---|---|
| `scale_count` | `int` | The number of scales: 1 for the tensor, or the number of channels |
| `symmetric` | `bool` | `True` learns only the step (LSQ); `False` also learns the zero point (LSQ+, INT formats only) |
| `init` | `LsqInit` | The start values: `"mse"` (least squared error, for activations), `"gauss"` (mean ± 3 standard deviations, for weights), `"lsq"` (the LSQ paper, symmetric only), or `"range"` (min and max) |

```python
quantizer = Quantizer(build_format("int4"), LsqScaler(init="mse"), batch_axis=0)
```

### forward

```python
scaler(x, number_format, granularity, update=True) -> tuple[torch.Tensor, torch.Tensor]
```

Returns the learned step and zero point, with their gradients. Both broadcast against `x`. On the first training batch, it sets the start values from `x`. The zero point is rounded to an integer, with a straight-through gradient.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values (the first batch sets the start values) |
| `number_format` | `NumberFormat` | The format |
| `granularity` | `Granularity` | How many scales |
| `update` | `bool` | `False` does not set the start values |

```python
torch.manual_seed(0)
scaler = LsqScaler()
scaler.train()
step, zero_point = scaler(torch.randn(1000), build_format("int8"), Granularity())
print(step)  # tensor(0.0320, grad_fn=...)  a parameter named "step"
```
