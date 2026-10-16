---
title: "group.py"
description: "Split tensors into groups that share one scale, and spread the scales back."
sidebar_position: 4
---

Splits a tensor into the groups that share one scale, for every granularity, and spreads one value for each group back over the tensor.

```python
from quaver.scales import group_count, group_merge, group_split
```

---

## group_count

```python
group_count(shape: torch.Size, granularity: Granularity) -> int
```

Returns how many scales a tensor of this shape gets.

| Argument | Type | Meaning |
|---|---|---|
| `shape` | `torch.Size` | The shape of the tensor |
| `granularity` | `Granularity` | The granularity |

```python
print(group_count(torch.Size([8, 64]), Granularity("block", axis=-1, block_size=16)))  # 32
```

---

## group_split

```python
group_split(x: torch.Tensor, granularity: Granularity) -> torch.Tensor
```

Reshapes `x` into (groups, values), so that each row shares one scale. For blocks, the size of the axis must be a multiple of `block_size`.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values |
| `granularity` | `Granularity` | The granularity |

```python
print(group_split(torch.arange(8.0).reshape(2, 4), Granularity("block", axis=-1, block_size=2)))
# tensor([[0., 1.],
#         [2., 3.],
#         [4., 5.],
#         [6., 7.]])
```

---

## group_merge

```python
group_merge(values: torch.Tensor, shape: torch.Size, granularity: Granularity) -> torch.Tensor
```

Spreads one value for each group back into a tensor that broadcasts against `shape`. One scale for the tensor gives a 0-dimensional value, and channels give a small shape.

| Argument | Type | Meaning |
|---|---|---|
| `values` | `torch.Tensor` | One value for each group (flat) |
| `shape` | `torch.Size` | The shape of the tensor |
| `granularity` | `Granularity` | The granularity |

```python
print(group_merge(torch.tensor([1.0, 2.0]), torch.Size([2, 3]), Granularity("channel", axis=0)))
# tensor([[1.],
#         [2.]])
```
