---
title: "Device math"
description: "The integer steps of each op, as a device must run them to match the integer mode bit for bit."
sidebar_position: 4
---

The integer mode of Quaver is the specification of the device: if your device runs these steps, its outputs match the integer mode bit for bit. The fixed numbers in them ($M_0$, $n$, zero points, the code of 6, block codes) are in `model.bin` ([Export to your device](../guides/export.md)).

Notation: $q$ is a code, $Z$ a zero point, $acc$ an accumulator, $M_0$ and $n$ a multiplier and a shift, and $\gg$ an arithmetic right shift. A negative $n$ means a left shift by $-n$.

## The steps of each op

| Op | Integer steps on the device |
|---|---|
| Quantize the input | $q = \mathrm{clamp}(\mathrm{round}(r / S) + Z)$: round first, then add the zero point. INT rounds half away from zero, FP half to even |
| Multiply-accumulate | $acc = \sum (q_x - Z_x)(q_w - Z_w) + b$, with the INT32 bias $b$. The sums are assumed to fit in INT32 |
| Block-scaled multiply-accumulate | $acc = \sum (\text{element code} \times \text{block code})$ of both sides, + bias. All blocks share one unit, so the products add up as they are. Up to 62 bits (54 bits for NVFP4 with 1024 inputs) |
| Quantize to block scales | block code = round(\|largest code of the block\| × $M_6 / 2^{n_6}$), half to even onto the block-scale format, at least 1. element = round(code × $M_r$ / (block code × $2^{n_r}$)), half to even. $M_6, n_6, M_r, n_r$ come from the fixed tensor scale |
| ReLU, ReLU6 | clamp the accumulator at the code of 0, and for ReLU6 also at the code of 6 (round(6 / S)) |
| Requantize | $q_y = \mathrm{clamp}\big(\big(((acc - Z) \cdot M_0 + 2^{n-1}) \gg n\big) + Z_y\big)$ (`"mul16_shift"`). An FP output rounds the exact value $acc \cdot M_0 / 2^n$ half to even onto its grid |
| Residual add | each input: $(q - Z) \cdot M_0$, brought to the larger of the two shifts $n$; add them; shift by $n$ once with rounding; add $Z_y$ and clamp |
| Residual add with a shared scale | $(q_a - Z) + (q_b - Z) + Z$, saturated at the ends of INT8 |
| Average pooling | the sum of the window ÷ the count, rounded half away from zero (half to even for FP) |
| NEQ | put the channels on the largest channel scale (fixed $M_0, n$); for each token, p = ⌊max\|v\| × $2^{20}$ / L1⌋ (the largest over the tokens for one scale per tensor); inverse = round(top × $2^{44}$ / p); $M_0, n$ of inverse / L1 by the run-time rule below; code = $(v \cdot M_0 + 2^{n+23}) \gg (n + 24)$ |
| Attention map (SimA) | the integer product Q Kᵀ times $M_0, n$ of $p_q \cdot p_k \cdot M_{0,H}$ (run-time rule), shifted right by $n + n_H + 40$ with rounding. $M_{0,H}, n_H$ are fixed: $H = (1/\sqrt{d}) / (\text{top}_q \cdot \text{top}_k \cdot s_A)$ |
| $M_0, n$ at run time | for a ratio a / b that changes with the data: $M_0$ = round(a × $2^n$ / b) (a half rounds up), with the largest $n$ that keeps $M_0$ in 16 bits; only bit counts, shifts, and an integer division |

## The requantization styles

| Style | $M_0$ | Rounding | Math on the device |
|---|---|---|---|
| `"mul16_shift"` (default) | up to $2^{16} - 1$ | half up | $(acc \cdot M_0 + 2^{n-1}) \gg n$; the product needs the bits of $acc$ + 16 |
| `"shift_only"` | 1 | half up | $(acc + 2^{n-1}) \gg n$; every scale ratio is a power of two |
| `"exact"` | none | half away | the real ratio in float64; for checks, not for a device |

CMSIS-NN and TFLite use a 31-bit multiplier and round twice, which can give a code one off from `"mul16_shift"` now and then. To match such a device, add its style to `REQUANTIZE_REGISTRY`.

## How this was checked

ResNet-8 was written again with integer multiplies, adds, shifts, and compares only (numpy int64, no float), using the constants from Quaver. On the 10,000 test images, every code of every layer, and the INT32 logits, matched the integer mode for all the quantized models, the NVFP4 model included. A model with SimA attention matched in 20 settings. [How the modes match bit for bit](../concepts/bit-exact.md) tells more.
