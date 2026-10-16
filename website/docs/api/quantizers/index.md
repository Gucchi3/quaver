---
title: "quantizers/"
description: "Quantizer, LSQ fake quantization, and building quantizers from QuantizerSpec."
sidebar_position: 1
---

The `Quantizer`, which puts a format, a scaler, and a granularity together, and the ways to build it from settings.

| File | What it holds |
|---|---|
| [quantizer.py](quantizer.md) | `Quantizer`: fake quantization (training mode) and codes (integer mode) |
| [lsq.py](lsq.md) | Fake quantization with a learned step, and its gradient scale |
| [build.py](build.md) | `build_quantizer()` and `SCALER_REGISTRY` |
| [types.py](types.md) | `QuantizerSpec` and the `Scaler` protocol |

It uses formats/ and scales/.
