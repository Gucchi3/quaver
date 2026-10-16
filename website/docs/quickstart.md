---
title: "Try it in 5 minutes"
description: "Make an INT8 conv layer, and check that the training mode and the integer mode give the same integer codes."
---

You will make one INT8 conv layer. Then you will check that its training mode and its integer mode give the same integers. You do not need a GPU. If you have not added Quaver to your project yet, see [Install](install.md) first.

Here is the whole code. It is the same as the Quick start in the README of the repository. Run it, and it prints `True`. The sections below explain each step.

The code imports `quaver` from the top. If you put Quaver in a subfolder, change the import to match, for example `from utils.quaver import ...`.

```python
import torch

from quaver import Granularity, QuantConv2d, Quantizer, RangeScaler, build_format, params_codes, precision_disable_tf32, switch_integer_mode

# 1. Turn off TF32, so that the two modes match exactly on a GPU
precision_disable_tf32()

# 2. Make an INT8 quantizer for the input, and a conv layer with an INT8 output
quantizer = Quantizer(build_format("int8"), RangeScaler(), Granularity(), batch_axis=0)
conv      = QuantConv2d(3, 16, 3, padding=1, output_format="int8")
images    = torch.randn(8, 3, 32, 32)

# 3. Training mode: measure the value ranges in one pass, and then freeze them with eval()
conv(*quantizer(images))
quantizer.eval()
conv.eval()
y, y_params = conv(*quantizer(images))

# 4. Integer mode: give integer codes, and get the same output codes as the training mode
switch_integer_mode(conv, True)
q_x, x_params = quantizer.encode(images)
q_y, _        = conv(q_x, x_params)
print(torch.equal(q_y, params_codes(y, y_params)))  # True
```

## Two switches with similar names

A Quaver layer has two switches. Their names are alike, but they do different things.

- **The Quaver mode** (the training mode or the integer mode) decides which math the layer runs. You change it with `switch_integer_mode()`.
- **PyTorch's `train()` and `eval()`** decide whether the scales keep changing. In `train()`, the layer measures the range of the values and updates its scales. In `eval()`, the scales stay fixed.

So the training mode works in both `train()` and `eval()`. This table shows when you use each pair.

| | `train()` | `eval()` |
|---|---|---|
| Training mode | You train the model with QAT. | You test the model during or after training. |
| Integer mode | You do not usually use this pair. | You check the model with the same math as your device. |

There is also a third mode, the float mode, with no quantization at all. Quaver uses it to measure the ranges of values before QAT. [Three modes](concepts/modes.md) explains all three.

## 1. Turn off TF32

On a GPU, call `precision_disable_tf32()` once, at the start of your program. Without it, the two modes may not match exactly. Keep it off during QAT too, so that the model you train is the same model your device runs.

By default, PyTorch lets cuDNN run float32 convolutions in TF32. TF32 keeps only 10 bits of the mantissa. The training mode computes the sums of a layer in float32, and Quaver turns them back into the exact INT32 sums that your device gets. With TF32, the sums lose bits, Quaver cannot get the exact INT32 sums back, and the training mode stops matching the integer mode. This setting applies to the whole process. It makes training about 10% slower on an RTX 3060. On a CPU, it changes nothing.

To run the code on a GPU, add `.to("cuda")` to `quantizer`, `conv`, and `images`. It prints `True` there too.

## 2. Make a quantizer and a layer

The code makes an INT8 quantizer for the input, and an INT8 conv layer.

| Argument of `Quantizer` | Meaning |
|---|---|
| `build_format("int8")` | The number format. Others include `"int4"` and `"fp4"`. |
| `RangeScaler()` | How the scale is set. In `train()`, the first batch sets the range of the values, and later batches update a moving average of it. In `eval()`, it uses that range. |
| `Granularity()` | How many scales there are. With no arguments, there is one scale for the whole tensor. |
| `batch_axis=0` | The batch axis. Give it to the quantizers of activations (the inputs and outputs of layers). |

`QuantConv2d` has INT8 weights, with one scale for each output channel. These scales are measured from the weights each time, so you do not set them. With `output_format="int8"`, it also quantizes its output to INT8.

Quaver layers always take and return a pair: the values and their [QuantParams](guides/quant-params.md). QuantParams holds the scale, the zero point, and the number format. In `conv(*quantizer(images))`, the `*` unpacks the pair from the quantizer, and gives the layer two arguments.

## 3. Run it in the training mode

A new layer starts in the training mode and in `train()`. In the first pass, the input quantizer and the output quantizer (inside the layer) measure the range of the values and set their scales. Then `eval()` fixes the scales, and the code runs the layer again. The scales must stay fixed from here on. If they changed between the two modes, the integer codes would not match.

The output `y` is a float tensor, but each value is an integer times the scale. This is called [fake quantization](learn/qat.md). In real training, you would compute the loss here and run backpropagation. See [Train the model](guides/training.md).

## 4. Check it in the integer mode

`switch_integer_mode(conv, True)` puts the layer in the integer mode. You can also give it a whole model, and it switches every Quaver layer inside. `quantizer.encode(images)` turns the input into integer codes (the integers that stand for the quantized values, as an int64 tensor). It returns the codes and their QuantParams. The layer needs both, so that it knows the scale of the input. In the integer mode, the layer takes integer codes and returns integer codes, using integer math only. It does a multiply-accumulate, and then requantizes the result with a multiplier and a shift. It also returns the QuantParams of its output, but this code does not need them, so it puts them in `_`.

`params_codes(y, y_params)` turns the output of the training mode into integer codes. The two sets of codes are the same, so the code prints `True`. The device is not in this test, but if your device runs the same steps, it gets the same output too. [Device math](api/device-math.md) lists these steps.

## Next steps

- Read the [Guides](guides/overview.md) in order. They build one program step by step, from your own layer to training and export.
- [Three modes](concepts/modes.md) explains the modes in more detail.
- To learn how quantization works, read [Quantization basics](learn/what-is-quantization.md).
