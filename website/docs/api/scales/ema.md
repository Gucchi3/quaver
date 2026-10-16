---
title: "ema.py"
description: "A moving average that starts from NaN, with no GPU sync."
sidebar_position: 5
---

The moving average that `RangeScaler` and `SharedScaler` use for their ranges.

```python
from quaver.scales import ema_update
```

---

## ema_update

```python
ema_update(running: torch.Tensor, current: torch.Tensor, momentum: float) -> torch.Tensor
```

Returns the moving average: `momentum × running + (1 − momentum) × current`. Where `running` is NaN (never measured), it returns `current` as it is. It uses `torch.where`, so there is no GPU sync.

| Argument | Type | Meaning |
|---|---|---|
| `running` | `torch.Tensor` | The average so far (NaN where nothing was measured yet) |
| `current` | `torch.Tensor` | The new measurement |
| `momentum` | `float` | How much of the old average to keep, such as 0.95 |

```python
print(ema_update(torch.tensor([2.0, float("nan")]), torch.tensor([3.0, 5.0]), 0.95))
# tensor([2.0500, 5.0000])
```
