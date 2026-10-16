---
title: "build.py"
description: "Build quantizers from QuantizerSpec, with a scaler chosen by name."
sidebar_position: 4
---

Builds a `Quantizer` from a `QuantizerSpec`. The scaler is chosen by name from `SCALER_REGISTRY`, so you can set it in a config file. [Write a quantizer](../../guides/quantizer.md#build-a-quantizer-from-settings) shows it in use.

```python
from quaver import build_quantizer
from quaver.quantizers import SCALER_REGISTRY, build_dynamic_scaler, build_lsq_scaler, build_max_scaler, build_nvfp4_scaler, build_percentile_scaler, build_shared_scaler
```

---

## build_quantizer

```python
build_quantizer(spec: QuantizerSpec, scale_count: int = 1) -> Quantizer
```

Builds a `Quantizer` from the settings in `spec`. An unknown scaler name raises `KeyError` with the list of names.

| Argument | Type | Meaning |
|---|---|---|
| `spec` | `QuantizerSpec` | The settings |
| `scale_count` | `int` | The number of scales for `"max"`, `"percentile"`, and `"lsq"`, such as the number of channels |

```python
spec      = QuantizerSpec(format="int4", scaler="lsq", granularity="channel", axis=0)
quantizer = build_quantizer(spec, scale_count=16)
print(type(quantizer.scaler).__name__, quantizer.scaler.step.shape)  # LsqScaler torch.Size([16])
```

---

## SCALER_REGISTRY

```python
SCALER_REGISTRY: dict[str, Callable[[QuantizerSpec, int], Scaler]]
```

The builders of the scalers, by name. To add a scaler, write `build_xxx_scaler(spec, scale_count)` that returns it, add it here, and set `QuantizerSpec.scaler` to its name.

| Name | Builder | Scaler |
|---|---|---|
| `"dynamic"` | `build_dynamic_scaler` | `DynamicScaler` |
| `"max"` | `build_max_scaler` | `RangeScaler` that follows the maximum |
| `"percentile"` | `build_percentile_scaler` | `RangeScaler` that follows a percentile |
| `"lsq"` | `build_lsq_scaler` | `LsqScaler` |
| `"shared"` | `build_shared_scaler` | a new `SharedScaler` |
| `"nvfp4"` | `build_nvfp4_scaler` | `BlockScaler("ue4m3")` |

```python
SCALER_REGISTRY["fixed"] = lambda spec, scale_count: FixedScaler(4.0)
quantizer = build_quantizer(QuantizerSpec(scaler="fixed"))
```

---

## build_dynamic_scaler

```python
build_dynamic_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler
```

Builds a `DynamicScaler`, which measures the tensor every time. It uses `spec.symmetric`.

| Argument | Type | Meaning |
|---|---|---|
| `spec` | `QuantizerSpec` | The settings |
| `scale_count` | `int` | Not used |

```python
scaler = build_dynamic_scaler(QuantizerSpec(symmetric=False), 1)
```

---

## build_max_scaler

```python
build_max_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler
```

Builds a `RangeScaler` that follows the maximum with a moving average. It uses `spec.momentum`, `spec.symmetric`, and `spec.freeze_step`.

| Argument | Type | Meaning |
|---|---|---|
| `spec` | `QuantizerSpec` | The settings |
| `scale_count` | `int` | The number of scales |

```python
scaler = build_max_scaler(QuantizerSpec(momentum=0.9), 1)
```

---

## build_percentile_scaler

```python
build_percentile_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler
```

Builds a `RangeScaler` that follows a percentile (`spec.percentile`, 99.9 by default) with a moving average, to leave out rare large values.

| Argument | Type | Meaning |
|---|---|---|
| `spec` | `QuantizerSpec` | The settings |
| `scale_count` | `int` | The number of scales |

```python
scaler = build_percentile_scaler(QuantizerSpec(percentile=99.0), 1)
```

---

## build_lsq_scaler

```python
build_lsq_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler
```

Builds an `LsqScaler` that learns the scale (LSQ), or also the zero point when `spec.symmetric` is false (LSQ+). It uses `spec.lsq_init` for the start values.

| Argument | Type | Meaning |
|---|---|---|
| `spec` | `QuantizerSpec` | The settings |
| `scale_count` | `int` | The number of scales |

```python
scaler = build_lsq_scaler(QuantizerSpec(symmetric=False, lsq_init="gauss"), 16)
```

---

## build_shared_scaler

```python
build_shared_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler
```

Builds a new `SharedScaler`. Each call makes a new one, so to share a scale, make one `SharedScaler` yourself and pass it to each quantizer.

| Argument | Type | Meaning |
|---|---|---|
| `spec` | `QuantizerSpec` | The settings |
| `scale_count` | `int` | Not used |

```python
scaler = build_shared_scaler(QuantizerSpec(), 1)
```

---

## build_nvfp4_scaler

```python
build_nvfp4_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler
```

Builds the two-level scaler of NVFP4: a UE4M3 scale for each block, and a tensor scale measured every time (`BlockScaler("ue4m3")`). Use it with `format="fp4"` and `granularity="block"`.

| Argument | Type | Meaning |
|---|---|---|
| `spec` | `QuantizerSpec` | The settings |
| `scale_count` | `int` | Not used |

```python
quantizer = build_quantizer(QuantizerSpec(format="fp4", scaler="nvfp4", granularity="block", axis=-1))
```
