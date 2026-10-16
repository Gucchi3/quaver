---
title: "Use the sample layers"
description: "Build a quantized CNN for CIFAR-10 from the sample layers of Quaver."
---

Quaver comes with sample layers that cover common models. On this page, you build the first version of the CIFAR-10 model from them.

## The sample layers

| Layer | What it does |
|---|---|
| `QuantConv2d` | Conv2d with quantized weights, an INT32 bias, an optional ReLU or ReLU6, and an optional requantization |
| `QuantLinear` | The same for a linear layer |
| `QuantAdd` | Adds two quantized tensors, such as a [residual](../learn/residual-add.md) |
| `QuantAvgPool2d` | [Average pooling](../learn/average-pooling.md) on codes |
| `QuantSimaAttention` | [SimA attention](../learn/sima-attention.md), which needs no softmax |
| `QuantBlockConv2d`, `QuantBlockLinear` | Conv2d and linear layers with [block scales](../learn/block-scales.md), such as NVFP4 |

The conv and linear layers take these settings:

| Argument | Meaning |
|---|---|
| `weight_format` | The format of the weights (`"int8"` by default). The weights get one scale for each output channel |
| `output_format` | The format to requantize the output to. With `None`, the layer returns its INT32 sums |
| `activation` | `"none"`, `"relu"`, or `"relu6"`. It runs on the sums, before the requantization |
| `style` | How your device requantizes: `"mul16_shift"` (default) or `"shift_only"` |
| `weight_symmetric`, `output_symmetric` | Pass `False` to give that quantizer a zero point |
| `output_scaler` | A scaler to share the output scale with other tensors |

## The first model

The model takes CIFAR-10 images (3 × 32 × 32) and returns 10 logits.

```python
from torch import nn

from quaver import Granularity, QuantAvgPool2d, QuantConv2d, QuantLinear, Quantizer, RangeScaler, build_format


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantizer    = Quantizer(build_format("int8"), RangeScaler(), Granularity(), batch_axis=0)
        self.conv1        = QuantConv2d(3, 32, 3, padding=1, output_format="uint8", activation="relu")
        self.conv2        = QuantConv2d(32, 64, 3, stride=2, padding=1, output_format="uint8", activation="relu")
        self.conv3        = QuantConv2d(64, 128, 3, stride=2, padding=1, output_format="uint8", activation="relu")
        self.pool         = QuantAvgPool2d(8)
        self.classifier   = QuantLinear(128, 10)
        self.integer_mode = False

    def forward(self, images):
        # Quantize the images: fake-quantized values in the training mode, and codes in the integer mode
        x, params = self.quantizer.encode(images) if self.integer_mode else self.quantizer(images)

        # Each layer takes the pair and returns the next pair
        x, params = self.conv1(x, params)
        x, params = self.conv2(x, params)
        x, params = self.conv3(x, params)
        x, params = self.pool(x, params)
        x, params = self.classifier(x.flatten(1), params)

        # The logits: real values in the training mode, and the INT32 codes with their QuantParams in the integer mode
        return (x, params) if self.integer_mode else x
```

A few points:

- After a ReLU, the values are never negative, so the outputs use `"uint8"`. This keeps all 256 codes for the positive values.
- The last layer has no `output_format`, so it returns the INT32 sums. They are good logits, and need no rounding.
- The model has its own `integer_mode` flag. `switch_integer_mode()` sets the flag of every module that has one, so the model knows when to give codes to its first layer.
- In the integer mode, the model returns the INT32 codes of the logits with their QuantParams, as your device would. `params_values(codes, params)` turns them into real values when you need them. The pair also tells the [export](export.md) the scale of the output.

## Load the weights of a float model

The sample layers keep the names of the weights of PyTorch layers (`weight` and `bias`). So if you have a float model with the same layer names, you can load its weights and start QAT from them:

```python
model = TinyNet()
model.load_state_dict(float_model.state_dict(), strict=False)  # the quantizers have their own state, so use strict=False
```

To fold BatchNorm into the conv weights first, use `fold_batch_norms()`.

Next, you will see what happens inside one of these layers.
