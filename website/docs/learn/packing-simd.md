---
title: "Packing and SIMD"
description: "Store several small codes in one memory word, and compute them at the same time."
---

Quantizing to 8 or 4 bits only pays off if the device stores and computes the codes in that size. Two techniques make this happen.

## Packing

Memory is read in words, often 32 bits wide. If each 8-bit code took a whole 32-bit word, nothing would be saved. So the codes are packed:

| Code size | Codes in one 32-bit word |
|---|---|
| 8 bits | 4 |
| 4 bits | 8 |
| 2 bits | 16 |

Each memory read now brings $32 / b$ codes, and the model takes $b / 32$ of the memory.

To use a packed code, the device shifts the word and masks the bits it wants. A signed code also needs **sign extension**: copying its top bit into the upper bits, so that, for example, the 4-bit code `1111` reads as −1, not 15.

## SIMD

SIMD (single instruction, multiple data) runs one operation on all the codes of a word at once. A 32-bit SIMD multiply-add on 8-bit codes does 4 products in one step. On 4-bit codes it does 8.

- With one multiplier, the device must unpack and multiply the codes one by one, so packing saves memory but not time.
- With 4 multipliers of 8 bits (or 8 of 4 bits), it computes a whole word in one step.

So the formats a device can compute in SIMD matter a lot when you choose the formats of a model. A 4-bit model is only faster than an 8-bit one if the hardware has 4-bit SIMD.

## FP4 on integer SIMD

FP4 codes can be computed on integer hardware too: each FP4 value is an integer times 0.5 ([INT and FP formats](int-fp.md)). A device can turn the 4-bit FP4 codes into small integers, and then use integer SIMD multipliers.
