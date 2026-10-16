---
title: "lsq.py"
description: "Helpers for training LSQ scales: their parameters, the optimizer groups, and a clamp."
sidebar_position: 6
---

Helpers for models with `LsqScaler`. [Train the model](../../guides/training.md#if-you-use-lsq) shows them in a training loop.

```python
from quaver import lsq_clamp_steps, lsq_param_groups
from quaver.model_tools import lsq_parameters
```

---

## lsq_parameters

```python
lsq_parameters(model: nn.Module) -> list[nn.Parameter]
```

Returns the learned steps and zero points of every `LsqScaler` in the model. A step exists only after the first training batch.

| Argument | Type | Meaning |
|---|---|---|
| `model` | `nn.Module` | The model |

```python
quantizer = Quantizer(build_format("int4"), LsqScaler(), batch_axis=0)
model     = nn.ModuleDict({"quantizer": quantizer, "linear": nn.Linear(4, 4)})
quantizer(torch.randn(8, 4))   # the first batch sets the step
print([p.shape for p in lsq_parameters(model)])  # [torch.Size([1])]
```

---

## lsq_param_groups

```python
lsq_param_groups(model: nn.Module, weight_decay: float) -> list[dict]
```

Makes the parameter groups for the optimizer: every other parameter with `weight_decay`, and the LSQ steps and zero points with no weight decay (decay would pull the scales toward 0). Call it after the first training batch, so the steps exist.

| Argument | Type | Meaning |
|---|---|---|
| `model` | `nn.Module` | The model |
| `weight_decay` | `float` | The weight decay of the other parameters |

```python
optimizer = torch.optim.AdamW(lsq_param_groups(model, weight_decay=1e-4), lr=2e-3)
# groups: the 2 parameters of the linear layer with weight decay 1e-4, and the LSQ step with 0
```

---

## lsq_clamp_steps

```python
lsq_clamp_steps(model: nn.Module) -> None
```

Pulls every LSQ step back above its lower limit (`STEP_MIN`). Call it after each `optimizer.step()`: Adam moves a step by about the learning rate, so one update can make a small step negative.

| Argument | Type | Meaning |
|---|---|---|
| `model` | `nn.Module` | The model |

```python
optimizer.step()
lsq_clamp_steps(model)
```
