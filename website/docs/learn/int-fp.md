---
title: "INT and FP formats"
description: "Evenly spaced codes or codes that are finer near 0: INT4 and FP4 side by side."
---

A number format decides which values the codes can stand for. INT formats space them evenly. FP formats space them finer near 0 and coarser far from 0.

## INT formats

An INT format with $b$ bits has evenly spaced codes:

| Format | Codes |
|---|---|
| INT8 | −128 to 127 |
| UINT8 | 0 to 255 |
| INT4 | −8 to 7 |
| UINT4 | 0 to 15 |

Every step is the same size $S$. This is [linear quantization](scale-zero-point.md).

## FP formats

An FP (floating point) format splits its bits into a sign, an exponent, and a mantissa, like a small version of FP32. FP4 in the E2M1 layout has 1 sign bit, 2 exponent bits, and 1 mantissa bit. Its values are:

$$
0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6 \quad \text{(and the same with a minus sign)}
$$

The steps grow with the size of the value: 0.5 near 0, and 2 near the top. UFP4 (E2M2, no sign bit, one more mantissa bit) has 16 values from 0 to 7:

$$
0,\ 0.25,\ 0.5,\ 0.75,\ 1,\ 1.25,\ 1.5,\ 1.75,\ 2,\ 2.5,\ 3,\ 3.5,\ 4,\ 5,\ 6,\ 7
$$

Like INT formats, an FP format is used with a scale: the real value is the format value times $S$.

## Why finer near 0 helps

The weights and activations of a model are usually bell-shaped: most values are near 0, and only a few are large. An FP format puts its fine steps where most values are, and its coarse steps where values are rare. So with the same number of bits, the average error is often smaller than with INT.

INT is better when the values are spread evenly over the range. At 8 bits, INT8 is usually good enough. At 4 bits, FP4 often wins. The [ResNet-8 example](../examples/resnet8.md) shows both.

## FP math with integers

An FP format looks like it needs float hardware, but it does not. Every FP4 value is a multiple of its smallest step, 0.5. So each value can be written as an integer times 0.5:

| FP4 value | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|---|---|---|
| Integer (value ÷ 0.5) | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 12 |

Products and sums of these integers are exact, so a device can compute FP4 layers with integer multipliers and adders. Quaver uses these integers as the codes of FP formats.

## Other non-linear formats

FP is one kind of non-linear quantization. Others include logarithmic formats (each step doubles), and formats learned from the data. Quaver supports INT and FP formats.
