---
title: "Inside a quantized layer"
description: "Read the forward pass of QuantConv2d, step by step."
---

Before you write a layer of your own, it helps to see how a sample layer works. `QuantConv2d` and `QuantLinear` share one forward pass, in `QuantLayer`. Here it is in short, with its steps.

```python
def forward(self, x, x_params):
    # 0. Pick the ops of the current mode (training, integer, or float)
    ops = mode_ops(self.integer_mode, self.float_mode)

    # 1. Quantize the weight: fake values in the training mode, codes in the integer mode
    quantize         = self.weight_quantizer.encode if self.integer_mode else self.weight_quantizer
    weight, w_params = quantize(self.weight)

    # 2. The sums, plus the INT32 bias
    bias, bias_codes = bias_quantize(self.bias, x_params, w_params, ...)
    if self.integer_mode:
        acc = self._layer_integer(x, x_params, weight, w_params) + bias_codes   # exact integer sums
    else:
        acc = self._layer_float(x, weight, bias)                                # float sums

    # 3. The scale of the sums: input scale × weight scale, for each output channel
    acc_params = QuantParams(accumulate_scale(x_params, w_params), zero, ACCUMULATOR_FORMAT)

    # 4. The activation, on the sums
    acc = ops.activation(acc, acc_params, self.activation)

    # 5. Requantize to the output format, with a multiplier and a shift
    out_params = self.output_quantizer.params(acc) if self.integer_mode else self.output_quantizer.measure(acc)
    return ops.requantize(acc, acc_params, out_params, self.style), out_params
```

## 0. Pick the ops of the mode

`mode_ops()` returns a set of ops for the current mode: the training version, the integer version, or the float version of each op. The layer calls `ops.activation()` and `ops.requantize()` without knowing the mode. So one forward pass covers all three modes.

## 1. Quantize the weight

The weight quantizer uses one scale for each output channel, measured from the weight each time. In the integer mode, `encode()` gives the codes. In the training mode, the quantizer gives fake-quantized values.

## 2. The sums and the bias

`_layer_integer()` and `_layer_float()` are the only parts that `QuantConv2d` and `QuantLinear` write themselves. The integer version adds up the products of the codes exactly. The float version runs the usual `F.conv2d()` on fake-quantized values.

The bias is quantized to INT32, with the scale of the sums (input scale × weight scale). This is how a device stores the bias, so that it can add it to the sums directly.

## 3. The scale of the sums

Each sum is (input code × weight code) added up, so its scale is the input scale times the weight scale. The weights have one scale for each output channel, so the sums do too.

## 4. The activation

ReLU and ReLU6 run on the sums, before the requantization. On codes, ReLU keeps the codes at or above the code of 0, and ReLU6 also cuts them at the code of 6.

## 5. Requantize

The output quantizer gives the scale of the output. In the training mode, it also watches the values and updates its range (`measure()`). In the integer mode, it only reads the scale (`params()`).

Then `ops.requantize()` maps the sums to the output codes. The ratio "sum scale ÷ output scale" becomes an integer multiplier and a shift, as on a device. In the training mode, `ops.requantize()` runs the same integer math on the float sums, and attaches the gradients. This is why the two modes [match bit for bit](../concepts/bit-exact.md).
