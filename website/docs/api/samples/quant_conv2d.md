---
title: "quant_conv2d.py"
description: "QuantConv2d: Conv2d with quantized weights, an INT32 bias, an optional ReLU, and an optional requantization."
sidebar_position: 3
---

The quantized Conv2d. It takes and returns a pair of a tensor and its `QuantParams`. [Use the sample layers](../../guides/sample-layers.md) builds a model with it.

```python
from quaver import QuantConv2d
```

---

## QuantConv2d

```python
QuantConv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, weight_format="int8", output_format=None, activation="none", style="mul16_shift", round_mode=None, weight_symmetric=True, output_symmetric=True, output_scaler=None)
```

Makes a Conv2d with quantized weights (one scale for each output channel), an INT32 bias, an optional ReLU or ReLU6, and an optional requantization of the output. It is not a subclass of `nn.Conv2d`, because its `forward()` takes and returns pairs. Its weight and bias have the names of `nn.Conv2d`, so the weights of a float model load into it.

| Argument | Type | Meaning |
|---|---|---|
| `in_channels` | `int` | The input channels |
| `out_channels` | `int` | The output channels |
| `kernel_size` | `int` | The size of the kernel |
| `stride` | `int` | The stride |
| `padding` | `int` | The padding |
| `bias` | `bool` | Whether the layer has a bias (stored as INT32) |
| `weight_format` | `str` | The format of the weights, one scale for each output channel |
| `output_format` | `str \| None` | The format to requantize the output to, or `None` to return the INT32 accumulator |
| `activation` | `ActivationKind` | `"none"`, `"relu"`, or `"relu6"`, on the accumulator before the requantization |
| `style` | `str` | How the device requantizes: `"mul16_shift"`, `"shift_only"`, or `"exact"` |
| `round_mode` | `RoundMode \| None` | The rounding rule, or `None` for the usual one of each format |
| `weight_symmetric` | `bool` | `False` gives the weights a zero point (INT formats only) |
| `output_symmetric` | `bool` | `False` gives the output a zero point (INT formats only) |
| `output_scaler` | `Scaler \| None` | A scaler for the output, such as a `SharedScaler` to share the scale; `None` makes a `RangeScaler` |

```python
conv = QuantConv2d(3, 16, 3, padding=1, output_format="int8", activation="relu")
```

### forward

```python
conv(x, x_params) -> tuple[torch.Tensor, QuantParams]
```

Runs the conv in the current mode, and returns the output and its parameters. In the integer mode, it takes codes and returns codes, the same codes as the training mode.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The input: fake-quantized values in the training mode, codes in the integer mode, float values in the float mode |
| `x_params` | `QuantParams` | The parameters of the input (one scale for the tensor) |

```python
quantizer = Quantizer(build_format("int8"), RangeScaler(), batch_axis=0)
images    = torch.randn(2, 3, 8, 8)
conv(*quantizer(images))                 # measure the ranges once
quantizer.eval(); conv.eval()
y, y_params = conv(*quantizer(images))   # training mode
print(y.shape, y_params.number_format.name)  # torch.Size([2, 16, 8, 8]) int8

switch_integer_mode(conv, True)
codes, _ = conv(*quantizer.encode(images))  # integer mode
print(torch.equal(codes, params_codes(y, y_params)))  # True
```
