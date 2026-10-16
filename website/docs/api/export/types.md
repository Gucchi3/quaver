---
title: "types.py"
description: "ExportLayout: the names of the layouts of the export."
sidebar_position: 5
---

The types of the export.

```python
from quaver.export import ExportLayout
```

---

## ExportLayout

```python
ExportLayout = Literal["channels_last", "channels_first", "hwc", "chw"]
```

The layouts that `model_export(layout=...)` takes. `"channels_last"` stores conv weights as (out, spatial…, in) and is the default; `"channels_first"` keeps the order of PyTorch, (out, in, spatial…). `"hwc"` and `"chw"` are their names for 2-D.

| Name | Weight of a Conv2d |
|---|---|
| `"channels_last"`, `"hwc"` | (out, h, w, in) |
| `"channels_first"`, `"chw"` | (out, in, h, w) |

```python
model_export(model, (image,), "out", layout={"classifier": "chw"})
```
