---
title: "minmax.py"
description: "Measure the range of the values in each group, by the maximum or a percentile."
sidebar_position: 3
---

Measures ranges. The scalers call it before they turn a range into a scale.

```python
from quaver.scales import minmax_measure
```

---

## minmax_measure

```python
minmax_measure(x: torch.Tensor, granularity: Granularity, percentile: float | None = None, symmetric: bool = True) -> tuple[torch.Tensor, torch.Tensor]
```

Measures the range of each group of `x` (one group for each scale), and returns the low ends and the high ends as flat tensors. Symmetric measures $|x|$ and returns (−high, high). With a percentile, the rare largest values are left out. It has no gradient.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values |
| `granularity` | `Granularity` | How the values are grouped |
| `percentile` | `float \| None` | A percentile such as 99.9, or `None` for the maximum |
| `symmetric` | `bool` | `True` for a range centered on 0 |

```python
x = torch.tensor([[-4.0, 1.0, 2.0], [0.5, -0.5, 3.0]])
print(minmax_measure(x, Granularity()))                                     # (tensor([-4.]), tensor([4.]))
print(minmax_measure(x, Granularity("channel", axis=0)))                    # (tensor([-4., -3.]), tensor([4., 3.]))
print(minmax_measure(x, Granularity("channel", axis=0), symmetric=False))   # (tensor([-4.0000, -0.5000]), tensor([2., 3.]))
```
