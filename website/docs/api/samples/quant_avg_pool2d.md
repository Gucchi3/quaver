---
title: "quant_avg_pool2d.py"
description: "QuantAvgPool2d: average pooling on integer codes."
sidebar_position: 9
---

The quantized average pooling. [Average pooling](../../learn/average-pooling.md) explains it.

```python
from quaver import QuantAvgPool2d
```

---

## QuantAvgPool2d

```python
QuantAvgPool2d(kernel_size, stride=None, round_mode=None)
```

Makes an average pooling on codes: add the codes of each window, divide by the count, and round. The output keeps the scale and the zero point of the input.

| Argument | Type | Meaning |
|---|---|---|
| `kernel_size` | `int` | The size of the window |
| `stride` | `int \| None` | The stride, or `None` for the window size |
| `round_mode` | `RoundMode \| None` | The rounding rule, or `None` for the usual one of the input format |

```python
pool = QuantAvgPool2d(8)   # a global pool over an 8×8 map
```

### forward

```python
pool(x, x_params) -> tuple[torch.Tensor, QuantParams]
```

Pools `x` in the current mode, and returns the output with `x_params` (the scale does not change).

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The input: fake-quantized values in the training mode, codes in the integer mode, float values in the float mode |
| `x_params` | `QuantParams` | The parameters of the input (one scale for the tensor) |

```python
y, y_params = pool(x, x_params)
print(y.shape)  # torch.Size([2, 128, 1, 1]) for x of shape (2, 128, 8, 8)
```
