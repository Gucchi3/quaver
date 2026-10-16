---
title: "types.py"
description: "QuantizerSpec, the settings to build a quantizer from a config file, and Scaler, the shape of every scaler."
sidebar_position: 5
---

The types of the quantizers: `QuantizerSpec` holds the settings of a quantizer, and `Scaler` describes what every scaler has.

```python
from quaver import QuantizerSpec
from quaver.quantizers import Scaler
```

---

## QuantizerSpec

```python
QuantizerSpec(format="int8", granularity="tensor", axis=0, block_size=16, scaler="max", symmetric=True, round_mode=None, momentum=0.95, percentile=99.9, pow2=False, lsq_init="mse", batch_axis=None, freeze_step=None)
```

Holds the settings to build a `Quantizer` from a config file. Copy the values of your config into it, and pass it to `build_quantizer()`. It is a frozen dataclass.

| Argument | Type | Meaning |
|---|---|---|
| `format` | `str` | The name of the number format, such as `"int8"` or `"fp4"` |
| `granularity` | `GranularityKind` | `"tensor"`, `"channel"`, or `"block"` |
| `axis` | `int` | The axis of the channels or the blocks |
| `block_size` | `int` | The size of a block |
| `scaler` | `str` | The name of the scaler in `SCALER_REGISTRY`: `"dynamic"`, `"max"`, `"percentile"`, `"lsq"`, `"shared"`, or `"nvfp4"` |
| `symmetric` | `bool` | `True` for a zero point of 0 |
| `round_mode` | `RoundMode \| None` | The rounding rule, or `None` for the usual one of the format |
| `momentum` | `float` | The momentum of the moving average (`"max"`, `"percentile"`, `"shared"`) |
| `percentile` | `float \| None` | The percentile of `"percentile"` |
| `pow2` | `bool` | `True` rounds the scale up to a power of two |
| `lsq_init` | `LsqInit` | The start values of `"lsq"` |
| `batch_axis` | `int \| None` | The axis of the samples; give 0 for activations |
| `freeze_step` | `int \| None` | Stop the range after this many updates (`"max"`, `"percentile"`, `"shared"`) |

```python
settings = {"format": "int8", "scaler": "percentile", "percentile": 99.9, "batch_axis": 0}
spec     = QuantizerSpec(**settings)
```

---

## Scaler

```python
class Scaler(Protocol):
    learnable: bool
    def __call__(self, x, number_format, granularity, update) -> tuple[torch.Tensor, torch.Tensor]: ...
```

Describes what every scaler has, so that you can write your own and pass it to `Quantizer`. A scaler has a `learnable` flag (`True` if its scale is trained by the loss), and a call that returns the scale and the zero point. Both must broadcast against `x`. With `update=False`, the call must not change any state.

| Argument of the call | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values |
| `number_format` | `NumberFormat` | The format |
| `granularity` | `Granularity` | How many scales |
| `update` | `bool` | Whether the scaler may update its state |

```python
from torch import nn

from quaver.scales import compute_scale, group_merge


class FixedScaler(nn.Module):
    learnable = False

    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def forward(self, x, number_format, granularity, update=True):
        # Always the range -limit to limit, whatever x is
        low, high = torch.tensor([-self.limit]), torch.tensor([self.limit])
        scale, zero_point = compute_scale(low, high, number_format, True)
        return group_merge(scale, x.shape, granularity), group_merge(zero_point, x.shape, granularity)


quantizer = Quantizer(build_format("int8"), FixedScaler(4.0))
```
