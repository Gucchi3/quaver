---
title: "calibrate.py"
description: "Run batches through the model in float, so the scalers learn the ranges before QAT or for PTQ."
sidebar_position: 5
---

Calibration: the scalers measure the ranges of the float values before training. [PTQ](../../learn/ptq.md) explains calibration.

```python
from quaver import calibrate_model
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

## calibrate_model

```python
calibrate_model(model: nn.Module, batches: Iterable, forward: Callable | None = None) -> None
```

Runs the batches through the model in the [float mode](../../concepts/modes.md#float-mode), without gradients: every layer computes in float (no rounding, no clipping), and every scaler records the ranges, like the observers of PyTorch. Then it turns quantization on again, and puts the model back in its former `train()` or `eval()` state. Use it for PTQ, or before QAT to start from good scales. LSQ steps get their start values from the first batch.

| Argument | Type | Meaning |
|---|---|---|
| `model` | `nn.Module` | The quantized model |
| `batches` | `Iterable` | The batches, such as a few batches of the training data |
| `forward` | `Callable \| None` | `forward(model, batch)`, when a batch is not one input tensor; `None` calls `model(batch)` |

```python
calibrate_model(model, [torch.randn(4, 3, 8, 8) for _ in range(3)])
print(model.quantizer.scaler.kept.flatten())  # the measured range, such as tensor([-4.0334,  4.0334])

# Batches of (images, labels): give a forward function
calibrate_model(model, train_loader, forward=lambda model, batch: model(batch[0].to(device)))
```
