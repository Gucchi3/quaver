---
title: "Three modes"
description: "The training mode, the integer mode, and the float mode: what each one is for, and how to switch."
---

Each Quaver layer can run in three modes. With them, one model and one set of weights cover the whole path: you measure the value ranges, train with QAT, and check the result with the same math as your device. One function call switches the mode of the whole model.

| Mode | Input | Math | You use it to |
|---|---|---|---|
| Training mode | fake-quantized values (float) | the same integer math as the integer mode, with float gradients | train with QAT |
| Integer mode | integer codes (int64) | integer math only, with exact sums | check the model, and make test vectors |
| Float mode | float values | float math, with no rounding | measure the value ranges before QAT |

## Training mode

This is the mode a new layer starts in. Its inputs and outputs are [fake-quantized](../learn/qat.md) values: floats that are an integer times the scale. So it fits into a normal PyTorch training loop.

Inside, a conv or linear layer computes its sums in float. Quaver turns these sums back into the exact integer sums, and runs every later step (the bias, the activation, and the requantization) with the same integer math as the integer mode. For the backward pass, it attaches the gradients of the float math. This is why the training mode and the integer mode give the same codes. [How the modes match bit for bit](bit-exact.md) explains the details.

## Integer mode

In the integer mode, each layer takes integer codes and returns integer codes, using integer math only. It runs the steps that your device runs, so its outputs match your device bit for bit. Use them to check your hardware, or keep them as test vectors.

```python
from quaver import switch_integer_mode

model.eval()                       # keep the scales fixed
switch_integer_mode(model, True)   # every Quaver layer in the model takes and returns codes
# ... run the model on codes ...
switch_integer_mode(model, False)  # back to the training mode
```

`switch_integer_mode()` sets the `integer_mode` flag of every module that has one. Your own model can have this flag too. Then its `forward()` can give the first layer codes in the integer mode, and fake-quantized values in the training mode.

## Float mode

In the float mode, the layers compute in float and round nothing. The quantizers still watch the values and update their scales, like the observers of PyTorch. The outputs are the same as those of the float model.

You seldom switch to this mode yourself. `calibrate_model()` turns it on, runs a few batches to set good scales, and turns it off again. Use it before QAT, or for [PTQ](../learn/ptq.md).

```python
from quaver import calibrate_model

calibrate_model(model, batches)    # float mode: measure the ranges, then back to the training mode
```

To switch it by hand, call `switch_quantizers(model, False)` to turn quantization off, and `switch_quantizers(model, True)` to turn it on again.

## Modes and train() or eval()

The mode and PyTorch's `train()` and `eval()` are two different switches. The mode decides which math runs. `train()` and `eval()` decide whether the scales keep changing. [Try it in 5 minutes](../quickstart.md#two-switches-with-similar-names) shows which pair to use when.

If you run the training step as a CUDA graph, capture it again after you switch the mode.
