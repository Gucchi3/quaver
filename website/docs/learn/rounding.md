---
title: "Rounding"
description: "How a value halfway between two codes is rounded, and why training and the device must round the same way."
mdx:
  format: mdx
---

import Widget from '@site/src/components/Widget';

Quantization rounds $r / S$ to an integer. Most values are clearly nearer to one code, but some fall exactly halfway, such as 2.5. Different rounding rules treat these halfway values differently.

## Rounding rules

| Rule | 2.5 | 3.5 | −2.5 | Used by |
|---|---|---|---|---|
| Half away from zero | 3 | 4 | −3 | TFLite, C's `round()` |
| Half to even | 2 | 4 | −2 | IEEE floats, `torch.round()`, NVFP4 |
| Half up | 3 | 4 | −2 | add 0.5, then take the floor: the cheap rule of integer hardware |
| Floor (truncate) | 2 | 3 | −3 | a plain right shift, with no rounding |

## Why floor is a bad idea

A plain right shift takes the floor: it always rounds down. Every value moves down by half a step on average, so the errors do not cancel out. They add up through the layers, and push the outputs in one direction.

The other three rules round to the nearest code, so their errors go up as often as down, and average out to about 0. On integer hardware, "half up" costs only one addition: add half a step ($2^{n-1}$ before a shift by $n$), then shift. [Requantization](requantization.md) uses this.

## Try it

Real values fall from the top, and the funnels send each one to the column of its nearest value. Switch the rounding to "floor" and watch the mean error turn negative. Change the format and the scale to see clipping and unused values.

<Widget src="quant-galton" params={{format: "int4", data: "sym"}} title="A Galton board of quantization" />

## Halfway values are rare, but not that rare

For values from a float computation, an exact halfway value is rare. But in integer math it happens often. For example, an average of 2 codes, or a sum shifted right by 1 bit, is a halfway value half of the time. So the rule matters in practice.

## Training and the device must agree

If training rounds halfway values to even, and the device rounds them up, then some codes differ by one. The model was trained for one rule, and runs with the other, so the accuracy on the device can be a little lower than in training. And you cannot check the device against the training model bit for bit.

## In Quaver

- INT formats round half away from zero by default, like TFLite.
- FP formats round half to even, like IEEE floats and NVFP4.
- Requantization rounds half up, with a multiplier and a shift, like integer hardware.

You can choose another rule with `round_mode` (`"half_away"`, `"half_even"`, `"half_up"`, or `"floor"`). The training mode and the integer mode always use the same rule, so they match bit for bit.
