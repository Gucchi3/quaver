---
title: "quant_layer.py"
description: "QuantLayer: the shared forward pass of the sample conv and linear layers, and the base class for your own."
sidebar_position: 2
---

`QuantLayer` holds the forward pass that `QuantConv2d` and `QuantLinear` share: the weight, the INT32 bias, the accumulator, the activation, and the requantization. Subclass it to make a conv-like layer of your own: [Write your own layer](../../guides/custom-layer.md) does it for a depthwise conv, and [Inside a quantized layer](../../guides/inside-a-layer.md) walks through `forward()`.

```python
from quaver.samples import QuantLayer
```

---

## QuantLayer

```python
QuantLayer(layer, weight_format, output_format, activation, style, round_mode, weight_symmetric, output_symmetric, output_scaler)
```

Takes the weight and the bias of a float layer, and makes the quantizers of the weight (one scale for each output channel, measured each time) and of the output. The weight and the bias keep the names of the float layer, so its `state_dict` loads as it is. A subclass sets `output_axis` and writes `_layer_float()` and `_layer_integer()`.

| Argument | Type | Meaning |
|---|---|---|
| `layer` | `nn.Conv2d \| nn.Linear` | A float layer, which gives the weight, the bias, and their start values |
| `weight_format` | `str` | The format of the weights |
| `output_format` | `str \| None` | The format to requantize the output to, or `None` for the INT32 accumulator |
| `activation` | `ActivationKind` | `"none"`, `"relu"`, or `"relu6"` |
| `style` | `str` | How the device requantizes |
| `round_mode` | `RoundMode \| None` | The rounding rule, or `None` for the usual one |
| `weight_symmetric` | `bool` | `False` gives the weights a zero point |
| `output_symmetric` | `bool` | `False` gives the output a zero point |
| `output_scaler` | `Scaler \| None` | A scaler for the output, or `None` for a `RangeScaler` |

```python
class QuantDepthwiseConv2d(QuantLayer):
    output_axis = 1

    def __init__(self, channels, kernel_size, stride=1, padding=0, output_format=None, activation="none"):
        conv = nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels)
        super().__init__(conv, "int8", output_format, activation, "mul16_shift", None, True, True, None)
        self.stride, self.padding, self.groups = stride, padding, channels

    def _layer_float(self, x, weight, bias):
        return F.conv2d(x, weight, bias, self.stride, self.padding, groups=self.groups)

    def _layer_integer(self, q_x, x_params, q_w, w_params):
        return accumulate_conv2d(q_x, x_params, q_w, w_params, self.stride, self.padding, 1, self.groups)
```

### forward

```python
layer(x, x_params) -> tuple[torch.Tensor, QuantParams]
```

Runs the layer in the current mode, and returns the output and its parameters: fake-quantized values in the training mode, integer codes in the integer mode (the same codes), and float values in the float mode. Without `output_format`, the output is the INT32 accumulator.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The input: fake-quantized values in the training mode, codes in the integer mode, float values in the float mode |
| `x_params` | `QuantParams` | The parameters of the input (one scale for the tensor) |

```python
y, y_params = layer(x, x_params)
```
