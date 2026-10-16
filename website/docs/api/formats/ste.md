---
title: "ste.py"
description: "Straight-through gradients (STE): round in the forward pass, and pass the gradient in the backward pass."
sidebar_position: 6
---

Straight-through gradients. Rounding has no useful gradient, so these functions round in the forward pass, and pass a gradient through in the backward pass. [QAT, STE, and LSQ](../../learn/qat.md) explains why.

```python
from quaver.formats import ste_attach, ste_quantize, ste_round
```

---

## ste_round

```python
ste_round(x: torch.Tensor, mode: RoundMode) -> torch.Tensor
```

Rounds `x` in the forward pass, and passes the gradient as it is (a slope of 1) in the backward pass.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values to round |
| `mode` | `RoundMode` | The rounding rule |

```python
x = torch.tensor([0.4, 1.6], requires_grad=True)
y = ste_round(x, "half_away")
y.sum().backward()
print(y, x.grad)  # tensor([0., 2.]) tensor([1., 1.])
```

---

## ste_quantize

```python
ste_quantize(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, number_format: NumberFormat, mode: RoundMode) -> torch.Tensor
```

Fake-quantizes `x` with the scale and the zero point: rounds it onto the format, and returns the real values. The gradient passes only for the values inside the range. The clipped values get 0. The scale and the zero point get no gradient. `Quantizer` uses it for scalers that do not learn.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The real values |
| `scale` | `torch.Tensor` | The scale |
| `zero_point` | `torch.Tensor` | The zero point |
| `number_format` | `NumberFormat` | The format |
| `mode` | `RoundMode` | The rounding rule |

```python
x = torch.tensor([0.3, 1.4, 9.0], requires_grad=True)
y = ste_quantize(x, torch.tensor(0.5), torch.tensor(0.0), build_format("int4"), "half_away")
y.sum().backward()
print(y, x.grad)  # tensor([0.5000, 1.5000, 3.5000]) tensor([1., 1., 0.])  (9.0 is clipped)
```

---

## ste_attach

```python
ste_attach(reference: torch.Tensor, value: torch.Tensor, inside: torch.Tensor | None = None) -> torch.Tensor
```

Returns `value` in the forward pass, exactly, and uses the gradient of `reference` in the backward pass. Where `inside` is `False`, the gradient is 0. The training-mode ops use it to give the exact results of their integer math the gradients of the float math.

| Argument | Type | Meaning |
|---|---|---|
| `reference` | `torch.Tensor` | The float result whose gradient is used |
| `value` | `torch.Tensor` | The exact result to return |
| `inside` | `torch.Tensor \| None` | Where the gradient passes (`True`), or `None` for everywhere |

```python
r = torch.tensor([1.0, 2.0], requires_grad=True)
y = ste_attach(r * 3, torch.tensor([5.0, 7.0]), torch.tensor([True, False]))
y.sum().backward()
print(y, r.grad)  # tensor([5., 7.]) tensor([3., 0.])
```
