---
title: "collect.py"
description: "Collect the quantizers and the scalers in a model, with their names."
sidebar_position: 2
---

Find the quantizers and the scalers inside a model.

```python
from quaver.model_tools import collect_quantizers, collect_scalers
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

## collect_quantizers

```python
collect_quantizers(model: nn.Module) -> list[tuple[str, Quantizer]]
```

Returns every `Quantizer` in the model, with its name, in the order of `model.named_modules()`.

| Argument | Type | Meaning |
|---|---|---|
| `model` | `nn.Module` | The model |

```python
print([name for name, _ in collect_quantizers(model)])
# ['quantizer', 'conv.weight_quantizer', 'conv.output_quantizer']
```

---

## collect_scalers

```python
collect_scalers(model: nn.Module, kind: type[T]) -> list[tuple[str, T]]
```

Returns every scaler of one class in the model, with its name. Pass `nn.Module` to get every scaler (every module with a `learnable` attribute).

| Argument | Type | Meaning |
|---|---|---|
| `model` | `nn.Module` | The model |
| `kind` | `type[T]` | The class of the scalers, such as `RangeScaler` or `LsqScaler`, or `nn.Module` for all |

```python
print([name for name, _ in collect_scalers(model, RangeScaler)])
# ['quantizer.scaler', 'conv.output_quantizer.scaler']
```
