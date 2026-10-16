---
title: "export/"
description: "Export a model in the integer mode for your device: model.json and a packed model.bin."
sidebar_position: 1
---

Export a trained model for your device. `model_export()` writes `model.json` (what to compute, layer by layer) and `model.bin` (every number, packed for the memory of the device). [From Quaver to your device](../../concepts/device-flow.mdx) explains the idea, and [Export to your device](../../guides/export.md) shows how to use it. The export needs numpy (the `export` extra).

| File | What it holds |
|---|---|
| [model_export.py](model_export.md) | `model_export()`, and the tables it uses: formats, roles, settings, layouts |
| [model_outputs.py](model_outputs.md) | `ModelOutputs`, which lets the export capture a model that returns codes with their parameters |
| [pack.py](pack.md) | `pack_codes()`, which packs codes into bytes as the device memory holds them |
| [types.py](types.md) | `ExportLayout`, the names of the layouts |

It uses `formats/` (the formats and the device ops), and the ops of the layers through `torch.export`.
