---
title: "switch.py"
description: "Turn quantization off and on (the float mode), and switch the integer mode."
sidebar_position: 3
---

Switch the mode of every layer of a model with one call. [Three modes](../../concepts/modes.md) explains the modes.

```python
from quaver import switch_integer_mode, switch_quantizers
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

## switch_integer_mode

```python
switch_integer_mode(model: nn.Module, enabled: bool) -> int
```

Puts every module that has an `integer_mode` flag into the integer mode (`True`), or back into the training mode (`False`). Your own model and layers can have the flag too. Returns how many modules changed. In the integer mode, the layers take and return integer codes, and compute like the device. Call `model.eval()` first, so the scales stay fixed.

| Argument | Type | Meaning |
|---|---|---|
| `model` | `nn.Module` | The model |
| `enabled` | `bool` | `True` for the integer mode, `False` for the training mode |

```python
model.eval()
print(switch_integer_mode(model, True))   # 2   the conv and the model itself
codes = model(torch.randn(1, 3, 8, 8))    # integer codes
switch_integer_mode(model, False)
```

---

## switch_quantizers

```python
switch_quantizers(model: nn.Module, enabled: bool) -> int
```

Turns quantization off (`False`) or on (`True`) in every `Quantizer`, and puts every layer with a `float_mode` flag into the float mode while it is off. In the float mode, the layers compute in float and the quantizers still update their scalers. Returns how many quantizers changed. `calibrate_model()` uses it. Capture CUDA graphs again after you switch.

| Argument | Type | Meaning |
|---|---|---|
| `model` | `nn.Module` | The model |
| `enabled` | `bool` | `False` turns quantization off (float mode), `True` turns it on |

```python
print(switch_quantizers(model, False), model.conv.float_mode)  # 3 True
switch_quantizers(model, True)
```
