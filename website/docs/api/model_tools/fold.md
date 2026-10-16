---
title: "fold.py"
description: "Fold each BatchNorm into the conv just before it."
sidebar_position: 7
---

BatchNorm folding: merge a BatchNorm into the conv before it, so the quantized model has no BatchNorm. [BatchNorm folding](../../learn/bn-folding.md) explains the math.

```python
from quaver import fold_batch_norms
```

---

## fold_batch_norms

```python
fold_batch_norms(model: nn.Module) -> nn.Module
```

Returns a copy of the model in `eval()`, with each BatchNorm folded into the conv just before it: the conv gets new weights and a new bias, and the BatchNorm becomes `nn.Identity`. The model itself does not change. It only folds a conv and a BatchNorm that are next to each other in the same parent module (Conv1d, Conv2d, and Conv3d with their BatchNorms). Fold the float model, and then load its weights into the quantized model.

| Argument | Type | Meaning |
|---|---|---|
| `model` | `nn.Module` | The float model |

```python
float_model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU())
folded      = fold_batch_norms(float_model)
print(folded)
# Sequential(
#   (0): Conv2d(3, 4, kernel_size=(3, 3), stride=(1, 1))
#   (1): Identity()
#   (2): ReLU()
# )
```
