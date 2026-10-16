---
title: "quant_linear.py"
description: "QuantLinear: Linear with quantized weights, an INT32 bias, an optional ReLU, and an optional requantization."
sidebar_position: 4
---

The quantized Linear. It works like `QuantConv2d`, on the last axis.

```python
from quaver import QuantLinear
```

---

## QuantLinear

```python
QuantLinear(in_features, out_features, bias=True, weight_format="int8", output_format=None, activation="none", style="mul16_shift", round_mode=None, weight_symmetric=True, output_symmetric=True, output_scaler=None)
```

Makes a Linear with quantized weights (one scale for each output feature), an INT32 bias, an optional ReLU or ReLU6, and an optional requantization. Without `output_format`, it returns the INT32 accumulator, which makes good logits for a classifier. Its weight and bias have the names of `nn.Linear`.

| Argument | Type | Meaning |
|---|---|---|
| `in_features` | `int` | The input features |
| `out_features` | `int` | The output features |
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
classifier = QuantLinear(32, 10)   # INT32 logits
```

### forward

```python
linear(x, x_params) -> tuple[torch.Tensor, QuantParams]
```

Runs the layer in the current mode on the last axis of `x`, and returns the output and its parameters.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The input: fake-quantized values in the training mode, codes in the integer mode, float values in the float mode |
| `x_params` | `QuantParams` | The parameters of the input (one scale for the tensor) |

```python
quantizer = Quantizer(build_format("int8"), RangeScaler(), batch_axis=0)
features  = torch.randn(4, 32)
classifier(*quantizer(features))
quantizer.eval(); classifier.eval()
logits, logit_params = classifier(*quantizer(features))
print(logits.shape, logit_params.number_format.name)  # torch.Size([4, 10]) int32
```
