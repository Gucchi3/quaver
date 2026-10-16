---
title: "model_tools/"
description: "Whole-model helpers: switch modes, calibrate, freeze the ranges, LSQ, fold BatchNorm, float settings, and syncs."
sidebar_position: 1
---

Helpers that work on a whole model at once.

| File | What it holds |
|---|---|
| [switch.py](switch.md) | `switch_integer_mode()` and `switch_quantizers()` |
| [calibrate.py](calibrate.md) | `calibrate_model()`: measure the ranges in float |
| [freeze.py](freeze.md) | `freeze_scalers()`: stop the range updates |
| [lsq.py](lsq.md) | Optimizer groups and a clamp for LSQ scales |
| [fold.py](fold.md) | `fold_batch_norms()` |
| [precision.py](precision.md) | Turn off TF32, and choose float32 or float64 math |
| [collect.py](collect.md) | Find the quantizers and the scalers of a model |
| [sync.py](sync.md) | `sync_count()`: count the GPU-to-CPU syncs |

It uses quantizers/ and scales/.
