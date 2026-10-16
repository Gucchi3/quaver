---
title: "Average pooling"
description: "Take the average of integer codes: add them up, divide by the window size, and round."
---

Average pooling replaces each window of values with their average. Many small models use a global average pool before the classifier.

## On codes

The average of real values $S\,(q_i - Z)$ is $S$ times the average of the codes, minus $Z$. So the scale and the zero point do not change, and the device only averages the codes:

$$
q_y = \mathrm{round}\left(\frac{1}{N} \sum_{i=1}^{N} q_i\right)
$$

$N$ is the number of values in the window, for example $8 \times 8 = 64$ for a global pool over an 8×8 map.

## Divide with integers

- The sum of the codes fits easily in 32 bits.
- If $N$ is a power of two, the division is a right shift. Add $N/2$ first to round to the nearest code ([Rounding](rounding.md)).
- If not, the device divides with an integer division, or multiplies by $1/N$ written as a multiplier and a shift.

## Rounding matters here

An average is exactly halfway between two codes quite often. For example, the average of 2 codes is a halfway value half of the time. So training and the device must use the same rule for halfway values. Otherwise many outputs differ by one code.

## In Quaver

`QuantAvgPool2d(kernel_size)` adds up the codes, divides by the window size, and rounds, with the same rule in both modes (half away from zero for INT formats, half to even for FP formats). The output keeps the scale and the zero point of the input.
