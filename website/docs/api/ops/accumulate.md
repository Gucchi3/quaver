---
title: "accumulate.py"
description: "Exact accumulators of conv, linear, and matrix products on integer codes: INT32, or wider with block scales."
sidebar_position: 2
---

The sums of products of a conv, a linear layer, or a matrix product, on integer codes. The sums are exact (int64), for any widths up to 62 bits. [Integer inference](../../learn/integer-inference.md) explains the accumulator.

```python
from quaver.ops import ACCUMULATOR_FORMAT, WIDE_ACCUMULATOR_FORMAT, accumulate_conv2d, accumulate_conv2d_device, accumulate_format, accumulate_linear, accumulate_linear_device, accumulate_matmul, accumulate_matmul_device, accumulate_product, accumulate_scale
```

The examples on this page use an INT8 input with the scale 0.1:

```python
x_params = QuantParams(torch.tensor(0.1), torch.tensor(0.0), build_format("int8"))
```

---

## ACCUMULATOR_FORMAT, WIDE_ACCUMULATOR_FORMAT

```python
ACCUMULATOR_FORMAT: NumberFormat        # int32
WIDE_ACCUMULATOR_FORMAT: NumberFormat   # int62
```

The formats of the accumulator: INT32 for normal layers, and a 62-bit format when a block-scaled tensor is in the sum (its block codes make the integers wider). The bias of a layer uses the format of its accumulator.

```python
print(ACCUMULATOR_FORMAT.name, WIDE_ACCUMULATOR_FORMAT.name)  # int32 int62
```

---

## accumulate_scale

```python
accumulate_scale(x_params: QuantParams, w_params: QuantParams) -> torch.Tensor
```

Returns the scale of the accumulator of a conv or linear layer, one for each output channel (flat): the step of the input integers × the step of the weight integers. The input must have one scale for the tensor, and the weight one for the tensor or one for each output channel.

| Argument | Type | Meaning |
|---|---|---|
| `x_params` | `QuantParams` | The parameters of the input |
| `w_params` | `QuantParams` | The parameters of the weight |

```python
w_params = QuantParams(torch.tensor([0.01, 0.02]).view(2, 1, 1, 1), torch.zeros(2, 1, 1, 1), build_format("int8"))
print(accumulate_scale(x_params, w_params))  # tensor([0.0010, 0.0020])
```

---

## accumulate_format

```python
accumulate_format(x_params: QuantParams, w_params: QuantParams) -> NumberFormat
```

Returns the format of the accumulator: `ACCUMULATOR_FORMAT` (int32), or `WIDE_ACCUMULATOR_FORMAT` (int62) when a block-scaled tensor is in the sum.

| Argument | Type | Meaning |
|---|---|---|
| `x_params` | `QuantParams` | The parameters of the input |
| `w_params` | `QuantParams` | The parameters of the weight |

```python
print(accumulate_format(x_params, w_params).name)  # int32
```

---

## accumulate_conv2d

```python
accumulate_conv2d(q_x: torch.Tensor, x_params: QuantParams, q_w: torch.Tensor, w_params: QuantParams, stride=1, padding=0, dilation=1, groups=1) -> torch.Tensor
```

Returns the exact accumulator (int64) of a conv on integer codes. The zero points are removed first, so the padding adds exact zeros. Block-scaled codes are multiplied by the codes of their block scales first. The bias is not included. Use it in `_layer_integer()` of your own conv layer ([Write your own layer](../../guides/custom-layer.md)).

| Argument | Type | Meaning |
|---|---|---|
| `q_x` | `torch.Tensor` | The input codes, (N, C, H, W) |
| `x_params` | `QuantParams` | Their parameters |
| `q_w` | `torch.Tensor` | The weight codes, (out, in / groups, kH, kW) |
| `w_params` | `QuantParams` | Their parameters |
| `stride` | `int \| tuple[int, int]` | The stride |
| `padding` | `int \| tuple[int, int]` | The padding |
| `dilation` | `int \| tuple[int, int]` | The dilation |
| `groups` | `int` | The number of groups (the number of channels for a depthwise conv) |

```python
q_x = torch.tensor([[[[1, 2], [3, 4]]]])
q_w = torch.tensor([[[[1, 0], [0, 1]]], [[[2, 2], [2, 2]]]])
print(accumulate_conv2d(q_x, x_params, q_w, w_params).flatten())  # tensor([ 5, 20])  1+4, and 2×(1+2+3+4)
```

---

## accumulate_linear

```python
accumulate_linear(q_x: torch.Tensor, x_params: QuantParams, q_w: torch.Tensor, w_params: QuantParams) -> torch.Tensor
```

Returns the exact accumulator (int64) of a linear layer on integer codes, without the bias.

| Argument | Type | Meaning |
|---|---|---|
| `q_x` | `torch.Tensor` | The input codes, (…, in) |
| `x_params` | `QuantParams` | Their parameters |
| `q_w` | `torch.Tensor` | The weight codes, (out, in) |
| `w_params` | `QuantParams` | Their parameters |

```python
w8 = QuantParams(torch.tensor(0.01), torch.tensor(0.0), build_format("int8"))
print(accumulate_linear(torch.tensor([[1, 2, 3]]), x_params, torch.tensor([[1, 1, 1], [2, 0, -1]]), w8))  # tensor([[ 6, -1]])
```

---

## accumulate_matmul

```python
accumulate_matmul(q_a: torch.Tensor, a_params: QuantParams, q_b: torch.Tensor, b_params: QuantParams) -> torch.Tensor
```

Returns the exact accumulator (int64) of the matrix product `a @ b` of integer codes, as in attention.

| Argument | Type | Meaning |
|---|---|---|
| `q_a` | `torch.Tensor` | The codes of `a` |
| `a_params` | `QuantParams` | Their parameters |
| `q_b` | `torch.Tensor` | The codes of `b` |
| `b_params` | `QuantParams` | Their parameters |

```python
print(accumulate_matmul(torch.tensor([[1, 2], [3, 4]]), x_params, torch.tensor([[5], [6]]), w8))  # tensor([[17], [39]])
```

---

## accumulate_conv2d_device, accumulate_linear_device, accumulate_matmul_device

```python
accumulate_conv2d_device(q_x, x_zero_point, x_block, q_w, w_zero_point, w_block, x_format, x_scale_format, w_format, w_scale_format, out_format, stride, padding, dilation, groups) -> torch.Tensor
accumulate_linear_device(q_x, x_zero_point, x_block, q_w, w_zero_point, w_block, x_format, x_scale_format, w_format, w_scale_format, out_format) -> torch.Tensor
accumulate_matmul_device(q_a, a_zero_point, a_block, q_b, b_zero_point, b_block, a_format, a_scale_format, b_format, b_scale_format, out_format) -> torch.Tensor
```

The device ops `accumulate_conv2d`, `accumulate_linear`, and `accumulate_matmul` (under `torch.ops.quaver`): the exact sum of (code − zero point) × (code of the block scale) products, as int64. The functions above call them with the parts of the `QuantParams`.

| Argument | Type | Meaning |
|---|---|---|
| `q_x`, `q_w` (`q_a`, `q_b`) | `torch.Tensor` | The codes of the two sides |
| `x_zero_point`, `w_zero_point` | `torch.Tensor` | Their zero points |
| `x_block`, `w_block` | `torch.Tensor \| None` | The codes of their block scales, or `None` without blocks |
| `x_format`, `w_format` | `str` | The names of their formats |
| `x_scale_format`, `w_scale_format` | `str` | The names of their block-scale formats (`""` without blocks) |
| `out_format` | `str` | The name of the accumulator format for the device: `"int32"` or `"int62"` |
| `stride`, `padding`, `dilation` | `list[int]` | The settings of the conv (conv only) |
| `groups` | `int` | The groups of the conv (conv only) |

```python
acc = torch.ops.quaver.accumulate_linear(torch.tensor([[1, 2, 3]]), torch.tensor(0), None, torch.tensor([[1, 1, 1], [2, 0, -1]]), torch.tensor(0), None, "int8", "", "int8", "", "int32")
print(acc)  # tensor([[ 6, -1]])
```

---

## accumulate_product

```python
accumulate_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
```

Returns `a @ b`. When one of the sizes is 1, it uses a broadcast product and a sum instead, since some CUDA builds of PyTorch compile a slow kernel for that case.

| Argument | Type | Meaning |
|---|---|---|
| `a` | `torch.Tensor` | The left matrix |
| `b` | `torch.Tensor` | The right matrix |

```python
print(accumulate_product(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0], [4.0]])))  # tensor([[11.]])
```
