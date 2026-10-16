---
title: "What is quantization?"
description: "Why we store and compute AI models with fewer bits, and what that costs."
---

A trained model holds millions of numbers: the weights, and the activations that flow between its layers. PyTorch stores each of them as a 32-bit float (FP32). Quantization stores them with fewer bits instead, such as 8-bit integers (INT8) or 4-bit numbers.

## Why fewer bits

| | FP32 | INT8 | INT4 |
|---|---|---|---|
| Bits for one number | 32 | 8 | 4 |
| 1 million weights | 4 MB | 1 MB | 0.5 MB |
| Numbers in one 32-bit memory read | 1 | 4 | 8 |

- **Less memory.** The model is 4 times smaller in INT8, and 8 times smaller in INT4. It can fit in the small memory of a microcontroller or an FPGA.
- **Less data to move.** Reading memory often costs more time and power than the math itself. With fewer bits, each read brings more numbers.
- **Cheaper math.** An 8-bit integer multiplier is much smaller and uses much less power than a 32-bit float multiplier. A chip can fit many of them side by side, and run them at once.

This is why quantization is used to run models on embedded devices, and also to make large language models cheaper to train and run.

## What it costs

With 8 bits, there are only 256 different values (with 4 bits, only 16). Every real number must be replaced by one of them, so it changes a little. These small changes add up through the layers, and the accuracy of the model can drop.

The whole art of quantization is to choose the values well, so that the model keeps its accuracy:

- **Which values are allowed?** Evenly spaced values ([linear quantization](scale-zero-point.md)), or values that are finer near 0 ([FP formats](int-fp.md)).
- **Where do they sit?** Centered on 0 ([symmetric](symmetric.md)), or moved to fit the data ([asymmetric](asymmetric.md)).
- **How wide is the range?** Wide enough for every value, or narrower to keep the steps fine ([choosing the range](clipping-range.md)).
- **How many ranges?** One for the whole tensor, or one for each channel or block ([granularity](granularity.md)).
- **When is it decided?** Once before inference ([static](static-dynamic.md)), or for each input (dynamic). With or without training again ([PTQ](ptq.md) and [QAT](qat.md)).

## Integer-only inference

On a device with integer hardware, quantization goes one step further: every layer computes with integers only, from the input image to the output. The device never needs a float. [Integer inference](integer-inference.md) and [requantization](requantization.md) show how.

The next page starts with the basic rule that turns a real number into an integer.
