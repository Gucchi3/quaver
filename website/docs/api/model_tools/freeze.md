---
title: "freeze.py"
description: "Stop or start again the range updates of the scalers, now or after a number of steps."
sidebar_position: 4
---

Stop the range updates of the activation scalers, after which their scales stay fixed. [Fast training](../../concepts/fast-training.md#stop-the-range-updates-on-the-gpu) shows why to stop them on the GPU.

```python
from quaver import freeze_scalers
```

The examples on this page use this small model:

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantizer    = Quantizer(build_format("int8"), RangeScaler(), batch_axis=0)
        self.conv         = QuantConv2d(3, 8, 3, padding=1, output_format="int8")
        self.integer_mode = False

    def forward(self, images):
        x, params = self.quantizer.encode(images) if self.integer_mode else self.quantizer(images)
        return self.conv(x, params)[0]


model = Net()
```

---

## freeze_scalers

```python
freeze_scalers(model: nn.Module, frozen: bool = True, step: int | None = None) -> int
```

Stops (`frozen=True`) or starts again (`frozen=False`) the range updates of every `RangeScaler` and `SharedScaler` in the model. With `step`, it instead makes them stop by themselves after that many updates; the count lives on the GPU, so the stop needs no sync, and a captured CUDA graph does not need to be captured again. Returns how many scalers it set. After you change `frozen`, capture CUDA graphs again.

| Argument | Type | Meaning |
|---|---|---|
| `model` | `nn.Module` | The model |
| `frozen` | `bool` | `True` stops the updates now, `False` starts them again |
| `step` | `int \| None` | Stop by itself after this many updates (training steps), or `None` |

```python
print(freeze_scalers(model))          # 2   both range scalers stop now
print(freeze_scalers(model, False))   # 2   they update again
freeze_scalers(model, step=5 * steps_per_epoch)   # call once before training: stop after 5 epochs
```
