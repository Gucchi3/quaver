---
title: "types.py"
description: "QuantParams, Granularity, BlockScale, and the names used by the scales."
sidebar_position: 7
---

The types of the scales. `QuantParams` travels with every quantized tensor, and `Granularity` says how many scales a tensor has. [Quantized values and QuantParams](../../guides/quant-params.md) shows how they are used.

```python
from quaver import Granularity, QuantParams
from quaver.scales import BlockScale, GranularityKind, LsqInit, MathDtype
```

---

## QuantParams

```python
QuantParams(scale, zero_point, number_format, block=None, peak=None)
```

Holds the scale, the zero point, and the number format of a quantized tensor. Every Quaver layer takes and returns a tensor together with its `QuantParams`. A code stands for the real value (code − zero point) × `code_scale`. It is a frozen dataclass.

| Argument | Type | Meaning |
|---|---|---|
| `scale` | `torch.Tensor` | The scale. One value, or one for each channel or block (it broadcasts against the tensor) |
| `zero_point` | `torch.Tensor` | The code of the real number 0 (0 for symmetric scales) |
| `number_format` | `NumberFormat` | The format of the codes |
| `block` | `BlockScale \| None` | The integer parts of the block scales, for block-scaled tensors (NVFP4 and the like) |
| `peak` | `torch.Tensor \| None` | The integer p of an NEQ output (SimA), which the device keeps |

```python
params = QuantParams(torch.tensor(0.25), torch.tensor(0.0), build_format("fp4"))
```

### code_scale

```python
params.code_scale -> torch.Tensor
```

The real value of one step of the code: `scale × step_min` of the format. For INT formats, this is the scale. For fp4, it is half the scale, since its codes count in steps of 0.5.

```python
print(params.code_scale)  # tensor(0.1250)
```

---

## Granularity

```python
Granularity(kind="tensor", axis=0, block_size=16)
```

Says how many scales a tensor gets. It is a frozen dataclass. [Granularity](../../learn/granularity.md) explains the choices.

| Argument | Type | Meaning |
|---|---|---|
| `kind` | `GranularityKind` | `"tensor"` (one scale), `"channel"` (one for each index of `axis`), or `"block"` (one for each `block_size` values along `axis`) |
| `axis` | `int` | The axis of the channels or the blocks |
| `block_size` | `int` | The number of values in a block. One block over the whole axis gives one scale for each token |

```python
Granularity()                                  # one scale for the tensor
Granularity("channel", axis=0)                 # one for each output channel of a weight
Granularity("block", axis=-1, block_size=16)   # one for each 16 values along the last axis
```

---

## BlockScale

```python
BlockScale(codes, scale_format, tensor_scale)
```

Holds the block scales of a block-scaled tensor as integers, for the exact integer math. A real value is: element code × `step_min` of its format × block scale × `tensor_scale`. `BlockScaler.split()` makes it, and `QuantParams.block` carries it.

| Argument | Type | Meaning |
|---|---|---|
| `codes` | `torch.Tensor` | The code of the block scale of each value, in `scale_format` (block scale = code × `step_min` of `scale_format`) |
| `scale_format` | `NumberFormat` | The format of the block scales, such as ue4m3 |
| `tensor_scale` | `torch.Tensor` | The second scale, for the whole tensor |

```python
weight                   = torch.randn(2, 32)
scale, zero_point, block = BlockScaler("ue4m3").split(weight, build_format("fp4"), Granularity("block", axis=-1, block_size=16), True)
print(block.codes.shape, block.scale_format.name)  # torch.Size([2, 32]) ue4m3
```

---

## GranularityKind, LsqInit, MathDtype

```python
GranularityKind = Literal["tensor", "channel", "block"]
LsqInit         = Literal["mse", "gauss", "lsq", "range"]
MathDtype       = Literal["float64", "float32"]
```

The names that some arguments take: the kind of a `Granularity`, the start values of an `LsqScaler`, and the float type of `precision_set_math()`.

| Name | Values |
|---|---|
| `GranularityKind` | `"tensor"`, `"channel"`, `"block"` |
| `LsqInit` | `"mse"` (least squared error), `"gauss"` (mean ± 3 standard deviations), `"lsq"` (the LSQ paper), `"range"` (min and max) |
| `MathDtype` | `"float64"`, `"float32"` |

```python
LsqScaler(init="gauss")
```
