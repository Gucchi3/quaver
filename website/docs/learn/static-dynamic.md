---
title: "Static and dynamic scales"
description: "Fix the scales before inference, or measure them for each input."
---

The scales of the weights are fixed once the model is trained. The scales of the activations are harder, because the activations change with every input. There are two ways to handle them.

| | Static | Dynamic |
|---|---|---|
| When the activation scales are set | before inference, once | during inference, for each input |
| Work at run time | none | measure the range, compute the scale |
| Fits inputs that differ a lot | less | better |
| Common on | embedded devices | servers, large language models |

## Static scales

The scales are measured on sample data ([calibration](ptq.md)) or during [QAT](qat.md), and then fixed. At inference, each layer uses its fixed scale. On a device, the scales turn into fixed multipliers and shifts, so the run-time math is only integer multiply, add, and shift.

The risk: if an input is very different from the data used to set the scales, its values may be clipped or rounded too coarsely.

## Dynamic scales

The device measures the range of each activation tensor while it runs, and computes its scale from it. The scale always fits the current input.

The cost: the device must find the largest value of the tensor before it can quantize it, which means one more pass over the data. It must also turn the new scale into a multiplier and a shift, which needs a division. This takes time and power, so embedded devices usually use static scales.

## Some layers need dynamic scales

Some values change a lot from input to input, such as the scores of attention. For them, a scale that is measured for each input can keep much more accuracy. SimA attention in Quaver does this with integer math only: the device finds the scale of each tensor with an integer division ([SimA attention](sima-attention.md)).

## In Quaver

- `RangeScaler` and `LsqScaler` give static scales after `eval()`.
- `DynamicScaler` measures each tensor every time. The sample layers use it for their weights. Since the weights are fixed, their scales are fixed too once the model is trained.
