---
title: "Symmetric quantization"
description: "A range centered on 0, with a zero point of 0: simple integer math, but some codes can go to waste."
---

Symmetric quantization fixes the zero point at 0, and makes the range the same size on both sides of 0.

## The rule

Find the largest absolute value $R$ of the data, and give it the largest code:

$$
R = \max(|r_{min}|,\ |r_{max}|), \qquad S = \frac{R}{q_{max}}, \qquad Z = 0
$$

Then quantizing and dequantizing need no zero point:

$$
q = \mathrm{clip}\left(\mathrm{round}\left(\frac{r}{S}\right),\ -q_{max},\ q_{max}\right), \qquad \hat{r} = S\,q
$$

For INT8, $q_{max} = 127$.

## Why it is popular

With $Z = 0$, a product of two codes needs no correction. A layer multiplies the input codes by the weight codes and adds them up, and the sum is simply the real sum divided by the two scales:

$$
\sum_i r_{x,i}\, r_{w,i} \approx S_x S_w \sum_i q_{x,i}\, q_{w,i}
$$

This keeps the integer math on the device small and fast. It is the usual choice for weights, because weights are usually spread evenly around 0.

## Its weak point

If the data is not centered on 0, some codes are never used. For example, take data from −400 to 1000:

- $R = 1000$, so $S = 1000 / 127 \approx 7.87$.
- The range is −1000 to 1000, but no value is below −400.
- The codes from −128 to −52 (about 30% of all codes) stand for values that never appear.

With fewer useful codes, the steps are coarser, and the rounding error is larger. [Asymmetric quantization](asymmetric.md) fixes this.

## Data that is never negative

The outputs of ReLU are never negative. A signed format would waste half of its codes on them. So use an **unsigned** format, still with $Z = 0$:

$$
S = \frac{r_{max}}{255}, \qquad q \in [0, 255] \quad \text{(UINT8)}
$$

This keeps all 256 codes for the real values, and the math stays as simple as above. The input images of a camera are like this too: their pixels are already UINT8 codes, with $S = 1/255$.

## −128 or −127?

INT8 codes go from −128 to 127, so a symmetric range has one extra code on the negative side. Many formats (such as TFLite for its weights) use only −127 to 127, so that the range is truly symmetric. Others use −128 to 127, and just never reach −128 from a symmetric scale. Both work, as long as training and the device agree.
