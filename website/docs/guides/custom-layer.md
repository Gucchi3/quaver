---
title: "Write your own layer"
description: "Add a depthwise conv on top of QuantLayer, and build an elementwise multiply from the ops."
---

There are two ways to write a quantized layer of your own.

- **On top of `QuantLayer`.** If your layer is a conv or a linear layer of some kind, write only its sums. `QuantLayer` does the rest: the weight, the bias, the activation, and the requantization.
- **From the ops.** For any other layer, write the forward pass yourself with the ops of Quaver. Then it runs in all three modes, and its two modes match bit for bit, just like the sample layers.

## On top of QuantLayer: a depthwise conv

A depthwise conv gives each channel its own small filter. Small models for devices, such as MobileNet, use it a lot. `QuantConv2d` has no `groups` argument, so here you add a depthwise conv.

```python
import torch.nn.functional as F

from torch import nn

from quaver.ops     import accumulate_conv2d
from quaver.samples import QuantLayer


class QuantDepthwiseConv2d(QuantLayer):
    output_axis = 1

    def __init__(self, channels, kernel_size, stride=1, padding=0, output_format=None, activation="none"):
        # The float layer gives the weight, the bias, and their start values
        conv = nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels)
        super().__init__(conv, "int8", output_format, activation, "mul16_shift", None, True, True, None)
        self.stride  = stride
        self.padding = padding
        self.groups  = channels

    def _layer_float(self, x, weight, bias):
        # Training and float modes: the float conv on fake-quantized values
        return F.conv2d(x, weight, bias, self.stride, self.padding, groups=self.groups)

    def _layer_integer(self, q_x, x_params, q_w, w_params):
        # Integer mode: the exact sums on codes (dilation 1, one group for each channel)
        return accumulate_conv2d(q_x, x_params, q_w, w_params, self.stride, self.padding, 1, self.groups)
```

`super().__init__()` takes the float layer, the weight format, the output format, the activation, the requantization style, the rounding (`None` for the usual one), whether the weight and the output are symmetric, and an output scaler (`None` for its own).

Now make the model a little like MobileNet. Replace `conv2` and `conv3` with a depthwise conv and a 1×1 conv each:

```python
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantizer    = Quantizer(build_format("int8"), RangeScaler(), Granularity(), batch_axis=0)
        self.conv1        = QuantConv2d(3, 32, 3, padding=1, output_format="uint8", activation="relu")
        self.depthwise2   = QuantDepthwiseConv2d(32, 3, stride=2, padding=1, output_format="uint8", activation="relu")
        self.pointwise2   = QuantConv2d(32, 64, 1, output_format="uint8", activation="relu")
        self.depthwise3   = QuantDepthwiseConv2d(64, 3, stride=2, padding=1, output_format="uint8", activation="relu")
        self.pointwise3   = QuantConv2d(64, 128, 1, output_format="uint8", activation="relu")
        self.pool         = QuantAvgPool2d(8)
        self.classifier   = QuantLinear(128, 10)
        self.integer_mode = False

    def forward(self, images):
        x, params = self.quantizer.encode(images) if self.integer_mode else self.quantizer(images)
        x, params = self.conv1(x, params)
        x, params = self.pointwise2(*self.depthwise2(x, params))
        x, params = self.pointwise3(*self.depthwise3(x, params))
        x, params = self.pool(x, params)
        x, params = self.classifier(x.flatten(1), params)
        return (x, params) if self.integer_mode else x
```

## From the ops: an elementwise multiply

An elementwise multiply of two tensors is used in gates and in squeeze-and-excitation blocks. It has no weight, so it does not fit `QuantLayer`. Write it from the ops:

```python
import torch

from torch import nn

from quaver     import Granularity, Quantizer, QuantParams, RangeScaler, build_format
from quaver.ops import ACCUMULATOR_FORMAT, mode_ops


class QuantMul(nn.Module):
    def __init__(self, output_format="int8", style="mul16_shift"):
        super().__init__()
        self.output_quantizer = Quantizer(build_format(output_format), RangeScaler(), Granularity(), batch_axis=0)
        self.style            = style
        self.integer_mode     = False
        self.float_mode       = False

    def forward(self, a, a_params, b, b_params):
        ops = mode_ops(self.integer_mode, self.float_mode)

        # The product: real values in the training mode, codes minus their zero points in the integer mode
        product = (a - a_params.zero_point) * (b - b_params.zero_point) if self.integer_mode else a * b

        # The product has the scale a_scale × b_scale, and no zero point
        scale          = a_params.code_scale * b_params.code_scale
        product_params = QuantParams(scale, torch.zeros_like(scale), ACCUMULATOR_FORMAT)

        # Requantize to the output format, like the device
        out_params = self.output_quantizer.params(product) if self.integer_mode else self.output_quantizer.measure(product)
        return ops.requantize(product, product_params, out_params, self.style), out_params
```

Three rules make a layer like this work in every mode:

1. **Give it the two flags**, `integer_mode` and `float_mode`, both `False` at the start. `switch_integer_mode()` and `switch_quantizers()` set them.
2. **Use `mode_ops()` for every step that rounds**, such as `ops.requantize()`. Then the training mode and the integer mode run the same integer math.
3. **Keep track of the scale** of each step in a `QuantParams`, as the product does here.

## Check your layer

Check that the two modes match, in the same way as [Try it in 5 minutes](../quickstart.md). Use inputs with a zero point that is not 0, so that a forgotten zero point shows up.

```python
from quaver import params_codes, switch_integer_mode

quantizer_a = Quantizer(build_format("int8"), RangeScaler(), Granularity(), batch_axis=0)
quantizer_b = Quantizer(build_format("uint8"), RangeScaler(symmetric=False), Granularity(), batch_axis=0)
mul         = QuantMul()
a, b        = torch.randn(8, 16, 5, 5) * 3, torch.rand(8, 16, 5, 5) * 2 - 0.3

# Measure the ranges once, and fix them
mul(*quantizer_a(a), *quantizer_b(b))
for module in (quantizer_a, quantizer_b, mul):
    module.eval()

# Compare the codes of the two modes
y, y_params = mul(*quantizer_a(a), *quantizer_b(b))
switch_integer_mode(mul, True)
q_y, _ = mul(*quantizer_a.encode(a), *quantizer_b.encode(b))
print(torch.equal(q_y, params_codes(y, y_params)))  # True
```

Also check that your layer has no syncs with `sync_count()` (see [Fast training](../concepts/fast-training.md)), so that it can run in a CUDA graph.
