---
title: "lsq.py"
description: "Fake quantization with a learned step (LSQ) and its gradient scale."
sidebar_position: 3
---

The math of LSQ: fake quantization whose step (and zero point) get gradients, and the scale of those gradients. `Quantizer` uses these for an `LsqScaler`. [QAT, STE, and LSQ](../../learn/qat.md) explains LSQ.

```python
from quaver.quantizers import lsq_grad_scale, lsq_quantize, lsq_scale_gradient
```

---

## lsq_quantize

```python
lsq_quantize(x: torch.Tensor, step: torch.Tensor, zero_point: torch.Tensor, number_format: NumberFormat, mode: RoundMode, grad_scale: float) -> torch.Tensor
```

Fake-quantizes `x` with a learned step (LSQ) and a learned zero point (LSQ+). Autograd gives the LSQ gradients: for the step, round(v) − v inside the range, and the end of the range outside. The gradients of the step and the zero point are multiplied by `grad_scale`.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The real values |
| `step` | `torch.Tensor` | The learned step (scale) |
| `zero_point` | `torch.Tensor` | The learned zero point (0 for LSQ) |
| `number_format` | `NumberFormat` | The format |
| `mode` | `RoundMode` | The rounding rule |
| `grad_scale` | `float` | The factor for the gradients of the step and the zero point |

```python
x    = torch.tensor([0.3, 1.4, 9.0], requires_grad=True)
step = torch.tensor(0.5, requires_grad=True)
y    = lsq_quantize(x, step, torch.tensor(0.0), build_format("int4"), "half_away", 1.0)
y.sum().backward()
print(y, x.grad, step.grad)  # tensor([0.5000, 1.5000, 3.5000]) tensor([1., 1., 0.]) tensor(7.6000)
```

---

## lsq_grad_scale

```python
lsq_grad_scale(count: int, number_format: NumberFormat) -> float
```

Returns the LSQ gradient scale $1 / \sqrt{N \cdot Q_p}$, where $N$ is the number of values that share one step and $Q_p$ is the largest value of the format. It keeps the step learning at a speed like the weights.

| Argument | Type | Meaning |
|---|---|---|
| `count` | `int` | The number of values that share one step (in one sample, for activations) |
| `number_format` | `NumberFormat` | The format |

```python
print(lsq_grad_scale(1000, build_format("int8")))  # 0.0028...  = 1 / sqrt(1000 × 127)
```

---

## lsq_scale_gradient

```python
lsq_scale_gradient(x: torch.Tensor, factor: float) -> torch.Tensor
```

Returns `x` with its value kept exactly, and only its gradient multiplied by `factor`. `Quantizer.measure()` uses it to put the LSQ gradient scale on a learned step.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The tensor |
| `factor` | `float` | The factor for its gradient |

```python
v = torch.tensor(2.0, requires_grad=True)
w = lsq_scale_gradient(v, 0.1)
(w * 3).backward()
print(w, v.grad)  # tensor(2.) tensor(0.3000)
```
