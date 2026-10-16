---
title: "model_outputs.py"
description: "ModelOutputs: let the export capture a model whose forward() returns codes with their parameters."
sidebar_position: 3
---

`torch.export` takes only tensors as outputs. `ModelOutputs` wraps a model so that pairs of codes and `QuantParams` become plain tensors, with a note of their parameters. `model_export()` uses it for you.

```python
from quaver.export import ModelOutputs
```

---

## ModelOutputs

```python
ModelOutputs(model)
```

Wraps a model for the export.

| Argument | Type | Meaning |
|---|---|---|
| `model` | `nn.Module` | The model, whose `forward()` returns a tensor, a pair, or a tuple or list of them |

```python
wrapped = ModelOutputs(model)
```

### forward

```python
wrapped(*inputs) -> tuple[torch.Tensor, ...]
```

Runs the model, and returns its outputs as a tuple of tensors: a tensor stays, and a pair (codes, `QuantParams`) gives its codes, with a note of their parameters during the export.

| Argument | Type | Meaning |
|---|---|---|
| `*inputs` | `torch.Tensor` | The inputs of the model |

```python
outputs = wrapped(torch.randn(1, 3, 8, 8))
print(type(outputs), outputs[0].dtype)  # <class 'tuple'> torch.int64
```

### pair

```python
ModelOutputs.pair(item) -> bool
```

Says whether `item` is a pair of codes and their `QuantParams`.

| Argument | Type | Meaning |
|---|---|---|
| `item` | `Any` | One output of the model |

```python
print(ModelOutputs.pair(model(torch.randn(1, 3, 8, 8))))  # True for a model that returns (codes, params)
```
