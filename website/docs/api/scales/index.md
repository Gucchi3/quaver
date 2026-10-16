---
title: "scales/"
description: "How scales are set: QuantParams, granularity, and the scalers (dynamic, range, LSQ, shared, block)."
sidebar_position: 1
---

Scales and their parameters: `QuantParams`, which travels with every quantized tensor, the granularity, and the scalers that set the scales.

| File | What it holds |
|---|---|
| [types.py](types.md) | `QuantParams`, `Granularity`, and `BlockScale` |
| [params.py](params.md) | Moving between values and codes, the integers of exact sums, and rounding in code units |
| [compute.py](compute.md) | Turning ranges into scales and zero points, MSE ranges, power-of-two and block scales |
| [minmax.py](minmax.md) | Measuring the range of each group |
| [group.py](group.md) | Splitting a tensor into the groups that share a scale |
| [ema.py](ema.md) | The moving average of the ranges |
| [dynamic_scaler.py](dynamic_scaler.md) | `DynamicScaler`: measure each tensor every time |
| [range_scaler.py](range_scaler.md) | `RangeScaler`: a moving average of the range, fixed after `eval()` |
| [shared_scaler.py](shared_scaler.md) | `SharedScaler`: one scale for several tensors |
| [lsq_scaler.py](lsq_scaler.md) | `LsqScaler`: a learned scale (LSQ, LSQ+) |
| [block_scaler.py](block_scaler.md) | `BlockScaler`: block scales such as NVFP4 |

It uses formats/.
