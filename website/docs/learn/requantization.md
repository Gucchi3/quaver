---
title: "Requantization"
description: "Rescale the INT32 sums to the next layer's codes with an integer multiplier and a shift: multiply, add, shift, and clamp."
---

After a layer adds up its products, the sums are INT32 numbers with a tiny scale. The next layer wants small codes, such as INT8, with a larger scale. Requantization moves the sums from one scale to the other, using integers only:

$$
q_y = \mathrm{clamp}\Big( \big((acc \cdot M_0 + 2^{n-1}) \gg n\big) + Z_y,\ q_{min},\ q_{max} \Big)
$$

The sections below build this rule one piece at a time.

## 1. Read the sum on a new ruler

The sum $acc$ has the scale $S_w S_x$, and the output has the scale $S_y$. The real value is the same on both rulers:

$$
y = S_w S_x \cdot acc = S_y \cdot q_y \qquad\Longrightarrow\qquad q_y = M \cdot acc, \quad M = \frac{S_w S_x}{S_y}
$$

$M$ is a real number, usually between 0 and 1. But the device has no floats.

## 2. Write M with integers

Write $M$ as an integer divided by a power of two:

$$
M \approx \frac{M_0}{2^n}
$$

Choose $n$ as large as possible while $M_0$ still fits in the multiplier ($b$ bits). For example, with $M = 0.002$ and a 16-bit multiplier:

- $0.002 \times 2^{24} = 33554.4$, which fits in 16 bits (up to 65535).
- $0.002 \times 2^{25} = 67108.9$, which does not.
- So $n = 24$ and $M_0 = 33554$. The error of $M$ is about $10^{-5}$.

$M_0$ and $n$ are computed once, before inference, from the fixed scales. With one scale for each output channel, there is a small table of $M_0$ and $n$, one pair for each channel.

## 3. Multiply, then shift

At run time, the device multiplies the sum by $M_0$, and divides by $2^n$ with a right shift:

| Step | Operation | Width |
|---|---|---|
| 1 | $P = acc \cdot M_0$ | $32 + b$ bits |
| 2 | $P + 2^{n-1}$ | $32 + b$ bits |
| 3 | $\gg n$ (arithmetic shift) | $32 + b - n$ bits |
| 4 | $+\,Z_y$ | |
| 5 | clamp to the output range | 8 bits |

Always multiply first and shift after. Shifting first would throw away the low bits of the sum before they count.

## 4. Round with half a step

A right shift alone takes the floor, which always rounds down and pushes every output down by half a step on average ([Rounding](rounding.md)). Adding $2^{n-1}$ first, which is half of $2^n$, turns the floor into rounding to the nearest code. This costs one addition.

## 5. Zero point and clamp

- Add the output zero point $Z_y$ (0 for symmetric outputs).
- Clamp the result to the range of the output format. Values outside **saturate** at the edge. Just keeping the low 8 bits would wrap around and flip the sign: 160 would become −96.
- ReLU comes almost for free here: clamp from below at $Z_y$ (the code of 0) instead of $q_{min}$.

## Other ways

Some devices use a different form. CMSIS-NN and TFLite use a 31-bit multiplier and round twice, and some simple hardware uses a shift only ($M_0 = 1$). Each form can give different codes for a few values, so training and the device must use the same one. Quaver uses a 16-bit multiplier and a shift by default (`"mul16_shift"`), or a shift only (`"shift_only"`).
