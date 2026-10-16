---
title: "formats/"
description: "Number formats (INT and FP), rounding, straight-through gradients, and the device ops."
sidebar_position: 1
---

The bottom layer: number formats and rounding. Every other folder uses it, and it uses none.

| File | What it holds |
|---|---|
| [format.py](format.md) | The formats by name (`build_format()`), and rounding values onto a format and turning them into codes |
| [fp.py](fp.md) | Small FP formats: FP4, UFP4, and UE4M3 |
| [integer.py](integer.md) | INT formats of any width |
| [round.py](round.md) | The exact rounding functions |
| [ste.py](ste.md) | Straight-through gradients for rounding and fake quantization |
| [types.py](types.md) | `NumberFormat` and `RoundMode` |
| [device.py](device.md) | Registering the integer steps of the device as torch ops (`torch.ops.quaver.*`) |

It uses nothing.
