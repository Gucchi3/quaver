---
title: "block_scaler.py"
description: "BlockScaler: the two levels of scales of block-scaled formats (NVFP4, and integer block scales)."
sidebar_position: 12
---

The scaler of block-scaled formats such as NVFP4: one small scale for each block, and one scale for the tensor. [Block scales](../../learn/block-scales.md) explains them.

```python
from quaver import BlockScaler
```

---

## BlockScaler

```python
BlockScaler(scale_format="ue4m3", tensor_scaler=None)
```

Makes the scaler of a block-scaled format. Each block gets a scale rounded to `scale_format`, and the tensor gets a second scale from `tensor_scaler`. Use it with `Granularity("block", axis, 16)`. The zero point is always 0.

| Argument | Type | Meaning |
|---|---|---|
| `scale_format` | `str` | The format of the block scales: `"ue4m3"` gives NVFP4, and `"uint6"` or `"uint10"` gives integer block scales. For integer devices, use `"uint6"`: its sums fit in 32 bits, while `"uint10"` and `"ue4m3"` need a wider accumulator ([why](../../guides/block-scale.md#the-width-of-the-accumulator)) |
| `tensor_scaler` | `nn.Module \| None` | The scaler that measures the range of the tensor. `None` uses `DynamicScaler()` (every time, for weights). A `RangeScaler()` keeps the tensor scale static, for activations |

```python
quantizer = Quantizer(build_format("fp4"), BlockScaler("ue4m3"), Granularity("block", axis=-1, block_size=16))
```

### forward

```python
scaler(x, number_format, granularity, update=True) -> tuple[torch.Tensor, torch.Tensor]
```

Returns the scale of each value (block scale × tensor scale) and a zero point of 0. Both have the shape of `x`.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values |
| `number_format` | `NumberFormat` | The format of the elements, such as fp4 |
| `granularity` | `Granularity` | The blocks |
| `update` | `bool` | `False` keeps a static tensor scale from updating |

```python
weight            = torch.randn(2, 32)
scale, zero_point = BlockScaler("ue4m3")(weight, build_format("fp4"), Granularity("block", axis=-1, block_size=16))
print(scale.shape)  # torch.Size([2, 32])
```

### split

```python
scaler.split(x, number_format, granularity, update) -> tuple[torch.Tensor, torch.Tensor, BlockScale]
```

Returns the scale of each value, the zero point 0, and the integer parts of the block scales (`BlockScale`: the code of each value's block scale, and the tensor scale). `Quantizer` uses it, so that the `QuantParams` of a block-scaled tensor carry its block scales for the exact integer math.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values |
| `number_format` | `NumberFormat` | The format of the elements |
| `granularity` | `Granularity` | The blocks |
| `update` | `bool` | `False` keeps a static tensor scale from updating |

```python
scale, zero_point, block = BlockScaler("ue4m3").split(weight, build_format("fp4"), Granularity("block", axis=-1, block_size=16), True)
print(block.scale_format.name)  # ue4m3
```

### tensor_scale

```python
scaler.tensor_scale(x, number_format, update) -> torch.Tensor
```

Returns the tensor scale: the largest absolute value from the tensor scaler ÷ (the top of the format × the largest block scale), so that the largest block gets the top block scale. Layers that quantize activations to blocks use it for their static tensor scale.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values |
| `number_format` | `NumberFormat` | The format of the elements |
| `update` | `bool` | `False` keeps a static tensor scale from updating |

```python
print(BlockScaler("ue4m3").tensor_scale(weight, build_format("fp4"), True))  # about 0.0013 for this weight
```
