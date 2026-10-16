---
title: "ops/"
description: "The quantized ops, each in a training, an integer, and a float version, with the integer steps of the device."
sidebar_position: 1
---

The quantized ops. Each op has three versions with the same arguments: `*_fake` (training mode), `*_integer` (integer mode), and `*_float` (float mode), and a `*_device` function with the integer steps that a device runs. The training and integer versions run the same integer math, so they give the same codes.

| File | What it holds |
|---|---|
| [accumulate.py](accumulate.md) | The exact sums of conv, linear, and matrix products |
| [bias.py](bias.md) | The INT32 bias |
| [requantize.py](requantize.md) | Requantization with a multiplier and a shift |
| [wide.py](wide.md) | Exact rounding of products wider than 64 bits |
| [block.py](block.md) | Quantizing to block scales (NVFP4) at run time |
| [activation.py](activation.md) | ReLU and ReLU6 on codes |
| [add.py](add.md) | The residual add, rounded once |
| [matmul.py](matmul.md) | The matrix product, for attention |
| [normalize.py](normalize.md) | NEQ and the attention map of SimA |
| [pool.py](pool.md) | Average pooling on codes |
| [mode.py](mode.md) | `mode_ops()`: pick the version of the ops for the mode |
| [types.py](types.md) | `RequantizeStyle`, `FixedPoint`, `OpSet`, and names |

It uses formats/ and scales/.
