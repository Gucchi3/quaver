---
title: "quant_block_conv2d.py"
description: "QuantBlockConv2d: Conv2d with block-scaled weights (NVFP4 and the like), with integer math only."
sidebar_position: 6
---

The block-scaled Conv2d. With fp4 weights and ue4m3 block scales, it is NVFP4. The NVFP4 model of the [ResNet-8 example](../../examples/resnet8.md) uses it.

```python
from quaver import QuantBlockConv2d
```

---

## QuantBlockConv2d

```python
QuantBlockConv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, weight_format="fp4", scale_format="ue4m3", input_format=None, output_format="int8", output_block=False, activation="none", style="mul16_shift", block_size=16, output_scaler=None)
```

Makes a Conv2d with block-scaled weights (a scale for each block of 16 input channels), optional block-scaled inputs, an exact wide accumulator, and the output in the format of the next op. `in_channels` must be a multiple of `block_size` (and `out_channels` too, with `output_block=True`).

| Argument | Type | Meaning |
|---|---|---|
| `in_channels` | `int` | The input channels |
| `out_channels` | `int` | The output channels |
| `kernel_size` | `int` | The size of the kernel |
| `stride` | `int` | The stride |
| `padding` | `int` | The padding |
| `bias` | `bool` | Whether the layer has a bias (stored with the wide accumulator) |
| `weight_format` | `str` | The format of the weight elements, such as `"fp4"` |
| `scale_format` | `str` | The format of the block scales: `"ue4m3"` for NVFP4, or `"uint6"` / `"uint10"`. Use `"uint6"` for integer devices: only its sums fit in 32 bits ([why](../../guides/block-scale.md#the-width-of-the-accumulator)) |
| `input_format` | `str \| None` | Quantize the input to blocks in this format here (such as `"ufp4"` after a ReLU), or `None` to take the input as it is (block-scaled, or INT8) |
| `output_format` | `str` | The format of the output |
| `output_block` | `bool` | `True` gives a block-scaled output, for the next block-scaled layer; `False` requantizes to one fixed scale |
| `activation` | `ActivationKind` | `"none"`, `"relu"`, or `"relu6"` |
| `style` | `str` | How the device requantizes |
| `block_size` | `int` | The number of input channels in a block |
| `output_scaler` | `Scaler \| None` | A scaler for the output (when `output_block` is `False`), such as a `SharedScaler` |

```python
conv = QuantBlockConv2d(32, 16, 3, padding=1, input_format="ufp4", activation="relu", output_format="int8")   # NVFP4 weights, UFP4 inputs
```

### forward

```python
conv(x, x_params) -> tuple[torch.Tensor, QuantParams]
```

Runs the conv in the current mode, and returns the output and its parameters.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The input: fake-quantized values in the training mode, codes in the integer mode, float values in the float mode |
| `x_params` | `QuantParams` | The parameters of the input (one scale for the tensor) |

```python
quantizer = Quantizer(build_format("uint8"), RangeScaler(), batch_axis=0)
x         = torch.rand(2, 32, 8, 8)
conv(*quantizer(x))
quantizer.eval(); conv.eval()
y, y_params = conv(*quantizer(x))
print(y.shape, y_params.number_format.name)  # torch.Size([2, 16, 8, 8]) int8
```
