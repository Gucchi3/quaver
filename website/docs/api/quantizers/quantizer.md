---
title: "quantizer.py"
description: "Quantizer: turn float tensors into quantized tensors, from a number format, a scaler, and a granularity."
sidebar_position: 2
---

`Quantizer` turns float tensors into quantized tensors. You build it from a number format, a scaler, and a granularity. [Write a quantizer](../../guides/quantizer.md) explains how to choose them.

```python
from quaver import Quantizer
```

---

## Quantizer

```python
Quantizer(number_format, scaler, granularity=Granularity(), round_mode=None, pow2=False, batch_axis=None)
```

Makes a quantizer from its parts. It is a `torch.nn.Module`, so it moves to the GPU with your model, and the state of its scaler (such as the measured range) is saved in the `state_dict`.

It raises `ValueError` if you give an FP format an asymmetric scaler (FP formats have no zero point), or turn on `pow2` with a `BlockScaler`.

| Argument | Type | Meaning |
|---|---|---|
| `number_format` | `NumberFormat` | The format of the codes, from `build_format()`, such as `build_format("int8")` |
| `scaler` | `Scaler` | How the scale is set: `DynamicScaler()`, `RangeScaler()`, `LsqScaler()`, `SharedScaler()`, or `BlockScaler()` |
| `granularity` | `Granularity` | How many scales: `Granularity()` for one, `Granularity("channel", axis=0)` for one per channel, and so on. One for the whole tensor by default |
| `round_mode` | `RoundMode \| None` | How to round: `"half_away"`, `"half_even"`, `"half_up"`, or `"floor"`. `None` uses the usual rule of the format (half away from zero for INT, half to even for FP) |
| `pow2` | `bool` | `True` rounds each scale up to a power of two, for a device that can only shift |
| `batch_axis` | `int \| None` | The axis of the samples (usually 0). Give it for activations, so that LSQ counts the values of one sample. Leave it `None` for weights |

```python
from quaver import Granularity, Quantizer, RangeScaler, build_format

quantizer = Quantizer(build_format("int8"), RangeScaler(), Granularity(), batch_axis=0)
```

### forward

```python
quantizer(x) -> tuple[torch.Tensor, QuantParams]
```

The training mode. Quantizes `x` and returns the fake-quantized values (float32) with their `QuantParams`. In `train()`, the scaler also looks at `x` and updates its range. The gradient passes through the rounding (STE), or trains the scale for `LsqScaler`.

When quantization is turned off (`switch_quantizers(model, False)`, as in calibration), it only updates the scaler, and returns `x` as it is.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The float tensor to quantize |

```python
x              = torch.tensor([[-1.0, -0.25, 0.1, 0.5, 2.0]])
values, params = quantizer(x)
# values: [[-1.0079, -0.2520, 0.0945, 0.5039, 2.0000]], params.scale: 0.0157 (= 2 / 127)
```

### encode

```python
quantizer.encode(x) -> tuple[torch.Tensor, QuantParams]
```

The integer mode. Returns the integer codes of `x` (int64) with their `QuantParams`. It does not change the scaler, so call it after `eval()`, when the scale is fixed. A code stands for the real value (code − zero point) × `params.code_scale`.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The float tensor to quantize |

```python
quantizer.eval()
codes, params = quantizer.encode(x)
# codes: [[-64, -16, 6, 32, 127]] (int64)
```

### params

```python
quantizer.params(x) -> QuantParams
```

Returns the current `QuantParams` for a tensor shaped like `x`, without quantizing and without changing the scaler. For most scalers, `x` only gives the shape. `DynamicScaler` and `BlockScaler` measure `x` itself. Layers use it in the integer mode to get the scale of their output.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | A tensor with the shape (and, for measuring scalers, the values) to get the parameters for |

```python
params = quantizer.params(torch.zeros(4, 5))
# params.scale: 0.0157
```

### measure

```python
quantizer.measure(x) -> QuantParams
```

Lets the scaler look at `x` (it updates its range in `train()`), and returns the `QuantParams`, without quantizing `x`. Layers use it in the training mode for an output that their own integer math rounds, such as the sums before requantization. With `LsqScaler`, the returned scale carries the LSQ gradient, so the layer's math can train it.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The float tensor for the scaler to look at |

```python
quantizer.train()
params = quantizer.measure(torch.tensor([[-3.0, 3.0]]))
# params.scale: 0.0161 (the moving average of the range moved from 2 toward 3)
```
