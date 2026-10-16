---
title: "quant_block_layer.py"
description: "QuantBlockLayer: the shared forward pass of the sample block-scaled conv and linear layers (NVFP4 and the like)."
sidebar_position: 5
---

`QuantBlockLayer` holds the forward pass that `QuantBlockConv2d` and `QuantBlockLinear` share: block scales on the weights (and the inputs), an exact wide accumulator, and the output in the format of the next op. Even the training mode runs the integer math here, since the accumulator is too wide for float values. [Block scales](../../learn/block-scales.md) explains the math.

```python
from quaver.samples import QuantBlockLayer
```

---

## QuantBlockLayer

```python
QuantBlockLayer(layer, weight_format, scale_format, input_format, output_format, output_block, activation, style, block_size, output_scaler)
```

Takes the weight and the bias of a float layer, and makes the quantizers of the weight blocks, of the input blocks (when `input_format` is set), and of the output. The input channels must be a multiple of `block_size`. Elements round half to even. A subclass writes `_layer_float()` and `_layer_integer()`, as for `QuantLayer`.

| Argument | Type | Meaning |
|---|---|---|
| `layer` | `nn.Conv2d \| nn.Linear` | A float layer, which gives the weight and the bias |
| `weight_format` | `str` | The format of the weight elements |
| `scale_format` | `str` | The format of the block scales |
| `input_format` | `str \| None` | Quantize the input to blocks in this format, or `None` |
| `output_format` | `str` | The format of the output |
| `output_block` | `bool` | Whether the output is block-scaled |
| `activation` | `ActivationKind` | `"none"`, `"relu"`, or `"relu6"` |
| `style` | `str` | How the device requantizes |
| `block_size` | `int` | The number of input channels in a block |
| `output_scaler` | `Scaler \| None` | A scaler for the output, or `None` |

```python
class QuantBlockLinearLike(QuantBlockLayer):
    output_axis = -1

    def _layer_float(self, x, weight, bias):
        return F.linear(x, weight, bias)

    def _layer_integer(self, q_x, x_params, q_w, w_params):
        return accumulate_linear(q_x, x_params, q_w, w_params)
```

### forward

```python
layer(x, x_params) -> tuple[torch.Tensor, QuantParams]
```

Runs the layer in the current mode: integer codes in the integer mode, fake values in the training mode (the same integer math), and float values in the float mode. The output is in the format of the next op, never the wide accumulator.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The input: fake-quantized values in the training mode, codes in the integer mode, float values in the float mode |
| `x_params` | `QuantParams` | The parameters of the input (one scale for the tensor) |

```python
y, y_params = layer(x, x_params)
```
