---
title: "shared_scaler.py"
description: "SharedScaler: one scale shared by several tensors in a forward pass, such as a residual stream."
sidebar_position: 10
---

A scaler that gives several tensors one scale, so that a device can add them as plain integers. [Shared scales](../../learn/shared-scales.md) explains the idea.

```python
from quaver import SharedScaler
```

---

## SharedScaler

```python
SharedScaler(percentile=None, momentum=0.95, symmetric=True, freeze_step=None)
```

Makes a scaler that one forward pass shares among several tensors. Give the same `SharedScaler` to every quantizer (or `output_scaler=` of every layer) that should share it. Call `begin_forward()` before the tensors, and `end_forward()` after them. The scale does not change in between. The moving average is updated once, in `end_forward()`, toward the widest range seen.

| Argument | Type | Meaning |
|---|---|---|
| `percentile` | `float \| None` | Follow a percentile such as 99.9, or `None` for the maximum |
| `momentum` | `float` | How much of the old range to keep at each update |
| `symmetric` | `bool` | `True` for a zero point of 0 |
| `freeze_step` | `int \| None` | Stop after this many updates (forward passes), or `None` to never stop |

```python
shared = SharedScaler()
add    = QuantAdd("int8", output_scaler=shared)
conv   = QuantConv2d(64, 64, 1, output_format="int8", output_scaler=shared)
```

### begin_forward

```python
scaler.begin_forward(x=None) -> None
```

Starts a forward pass: forgets the ranges seen so far. If the range was never set, `x` sets it, so even the first pass uses one scale. Pass the first tensor of the stream.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor \| None` | The first tensor of the stream, to set a range that was never set |

```python
shared.train()
shared.begin_forward(torch.tensor([-1.0, 1.0]))
```

### forward

```python
scaler(x, number_format, granularity, update=True) -> tuple[torch.Tensor, torch.Tensor]
```

Returns the shared scale and zero point, and remembers the range of `x` for `end_forward()`. The quantizers call this for you.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | One of the tensors that share the scale |
| `number_format` | `NumberFormat` | The format |
| `granularity` | `Granularity` | Must be one scale for the tensor |
| `update` | `bool` | `False` does not remember the range of `x` |

```python
int8 = build_format("int8")
print(shared(torch.tensor([-1.0, 1.0]), int8, Granularity()))  # (tensor(0.0079), tensor(0.))
print(shared(torch.tensor([-4.0, 2.0]), int8, Granularity()))  # (tensor(0.0079), tensor(0.))  the same scale
```

### end_forward

```python
scaler.end_forward() -> None
```

Finishes a forward pass: moves the kept range toward the widest range seen in it. After `freeze_step` updates, the range stays as it is.

```python
shared.end_forward()
print(shared.kept.flatten())  # tensor([-1.1500,  1.1500])  0.95 × 1 + 0.05 × 4
```
