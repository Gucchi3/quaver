---
title: "Choosing the range"
description: "How wide the range should be: the balance between rounding error and clipping error, and the common ways to choose it."
---

The scale sets the range of a quantized tensor: with INT8 and a symmetric scale $S$, the range is $\pm 127\,S$. The edge of the range is often called the **clipping value** $\alpha$. How to choose it is one of the most important choices in quantization.

## The balance

- **A wide range** keeps every value, even rare large ones (outliers). But the steps are coarse, so every value gets a larger rounding error.
- **A narrow range** gives fine steps and small rounding errors. But the values outside it are clipped to the edge, and each of them gets a large clipping error.

For example, say most values are between −1 and 1, but one value is 10:

| Range | Step (INT8) | Error of a typical value | Error of the outlier |
|---|---|---|---|
| ±10 (keep the outlier) | 0.079 | up to 0.039 | 0 |
| ±1 (clip the outlier) | 0.0079 | up to 0.0039 | 9 |

Clipping one outlier makes every other value 10 times more exact. Whether that is worth it depends on how much the outlier matters to the model.

## Common ways to choose the range

| Method | How | Notes |
|---|---|---|
| Min-max | Use the smallest and largest values seen | Simple, but one outlier can stretch the range |
| Moving average of min-max | Average the ranges of many batches | Smooths out batches with rare values; common for activations |
| Percentile | Ignore the top 0.1% (or 1%) of the values | Clips rare outliers on purpose |
| MSE | Search for the range with the smallest squared error | Used in calibration |
| PACT | Learn the clipping value of ReLU with the loss | An early learned method |
| LSQ | Learn the scale itself with the loss | Common in QAT; see [QAT](qat.md) |

The first four look at the values. The last two let the training loss decide, so the range is the one that is best for the accuracy of the model, not for the error of each value.

## Weights and activations

- **Weights** are known after training, so their range can be measured exactly. Min-max per output channel works well for them.
- **Activations** change with every input. Their range must be estimated from many inputs (calibration), averaged during training, or learned.

## In Quaver

| Scaler | Method |
|---|---|
| `DynamicScaler()` | Min-max of each tensor, every time (or a percentile) |
| `RangeScaler()` | Moving average of min-max (or a percentile) |
| `LsqScaler()` | LSQ, or LSQ+ with a zero point |
