---
title: "range_scaler.py"
description: "RangeScaler: follow the range with a moving average while training, and keep it for inference."
sidebar_position: 9
---

The usual scaler for activations. It follows the range of the values with a moving average while training, and keeps it fixed after `eval()`.

```python
from quaver import RangeScaler
```

---

## RangeScaler

```python
RangeScaler(percentile=None, momentum=0.95, scale_count=1, symmetric=True, freeze_step=None)
```

Makes a scaler that keeps a moving average of the range. The range starts as NaN (not measured), and the first training batch sets it. It can stop by itself after some updates (`freeze_step`), with no GPU sync, so a captured CUDA graph does not need to be captured again. Setting `scaler.frozen = True` stops it at once.

| Argument | Type | Meaning |
|---|---|---|
| `percentile` | `float \| None` | Follow a percentile such as 99.9, or `None` for the maximum |
| `momentum` | `float` | How much of the old range to keep at each update |
| `scale_count` | `int` | The number of scales: 1 for the tensor, or the number of channels |
| `symmetric` | `bool` | `True` for a zero point of 0; `False` for a zero point (INT formats only) |
| `freeze_step` | `int \| None` | Stop after this many updates (training steps), or `None` to never stop |

```python
quantizer = Quantizer(build_format("int8"), RangeScaler(freeze_step=5 * steps_per_epoch), batch_axis=0)
```

### forward

```python
scaler(x, number_format, granularity, update=True) -> tuple[torch.Tensor, torch.Tensor]
```

In `train()` (and with `update=True`), moves the kept range toward the range of `x`. Then returns the scale and the zero point from the kept range. In `eval()`, the range stays fixed. Before the first training batch, the range is NaN, so train or run `calibrate_model()` first.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values |
| `number_format` | `NumberFormat` | The format |
| `granularity` | `Granularity` | How many scales |
| `update` | `bool` | `False` reads the scale without updating |

```python
scaler, int8 = RangeScaler(), build_format("int8")
scaler.train()
print(scaler(torch.tensor([-1.0, 2.0]), int8, Granularity()))  # (tensor(0.0157), tensor(0.))  the range is 2
print(scaler(torch.tensor([-3.0, 3.0]), int8, Granularity()))  # (tensor(0.0161), tensor(0.))  0.95 × 2 + 0.05 × 3 = 2.05
scaler.eval()
print(scaler(torch.tensor([100.0]), int8, Granularity()))      # (tensor(0.0161), tensor(0.))  fixed
```
