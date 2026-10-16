---
title: "sync.py"
description: "Count the GPU-to-CPU syncs of a function."
sidebar_position: 9
---

Check that your own layer makes no GPU-to-CPU syncs, so that it can run in a CUDA graph. [Fast training](../../concepts/fast-training.md#no-syncs-in-a-training-step) explains why.

```python
from quaver.model_tools import sync_count
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

## sync_count

```python
sync_count(fn: Callable, *args: object) -> int
```

Runs `fn(*args)` once, and returns the number of times it made the CPU wait for the GPU (for example with `.item()`). A CUDA graph needs 0. It returns 0 when there is no GPU.

| Argument | Type | Meaning |
|---|---|---|
| `fn` | `Callable` | The function or module to run |
| `*args` | `object` | Its arguments |

```python
model = model.cuda()
print(sync_count(model, torch.randn(4, 3, 8, 8, device="cuda")))           # 0
print(sync_count(lambda t: t.sum().item(), torch.ones(3, device="cuda")))  # 1   .item() waits for the GPU
```
