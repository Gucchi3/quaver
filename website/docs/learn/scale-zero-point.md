---
title: "Scale and zero point"
description: "Linear quantization: the rule that maps real numbers to integer codes, and back."
---

The most common way to quantize is linear quantization. It puts evenly spaced marks on the real number line, like the marks on a ruler, and gives each mark an integer, called its **code**.

Two numbers describe the ruler:

| Name | Symbol | Meaning |
|---|---|---|
| Scale | $S$ | The real distance between two marks: how much one step of the code is worth |
| Zero point | $Z$ | The code of the real number 0 |

## Quantize and dequantize

To **quantize** a real number $r$, divide it by the scale, round it to the nearest integer, add the zero point, and keep it inside the range of the format:

$$
q = \mathrm{clip}\left(\mathrm{round}\left(\frac{r}{S}\right) + Z,\ q_{min},\ q_{max}\right)
$$

To **dequantize** a code $q$ back to a real number:

$$
\hat{r} = S\,(q - Z)
$$

$q_{min}$ and $q_{max}$ are the smallest and largest codes of the format: −128 and 127 for INT8, 0 and 255 for UINT8.

## An example

Take INT8 with $S = 0.1$ and $Z = 0$:

| Real number $r$ | $r / S$ | Code $q$ | Back to real $\hat{r}$ | Error |
|---|---|---|---|---|
| 0.537 | 5.37 | 5 | 0.5 | 0.037 |
| −2.46 | −24.6 | −25 | −2.5 | 0.04 |
| 20.0 | 200 | 127 (clipped) | 12.7 | 7.3 |

## Two kinds of error

- **Rounding error.** Inside the range, $\hat{r}$ is the nearest mark, so the error is at most half a step, $S/2$. A smaller scale gives finer marks and a smaller error.
- **Clipping error.** A number outside the range sticks to the end of the range, like 20.0 above. This error can be large.

A smaller scale cuts the rounding error, but makes the range narrower, so more numbers are clipped. Choosing the scale is a balance between the two. [Choosing the range](clipping-range.md) covers this.

## Why the zero point is an integer

The zero point is an integer code, so the real number 0 is exactly a mark. This matters, because 0 is everywhere in a model: the padding around an image, the outputs of ReLU, and so on. If 0 were between two marks, every padded 0 would turn into a small error.

## Symmetric or asymmetric

There are two common ways to place the ruler:

- [Symmetric quantization](symmetric.md) puts the zero point at 0, so the range is centered on 0.
- [Asymmetric quantization](asymmetric.md) moves the zero point, so the range fits the data.
