---
title: "bias.py"
description: "Quantize the bias to the scale of the accumulator (INT32, or wide for block-scaled tensors)."
sidebar_position: 3
---

The bias of a layer is stored as an integer with the scale of the accumulator, so that a device can add it to the sums directly.

```python
from quaver.ops import bias_quantize
```

---

## bias_quantize

```python
bias_quantize(bias: torch.Tensor, x_params: QuantParams, w_params: QuantParams, mode: RoundMode) -> tuple[torch.Tensor, torch.Tensor]
```

Quantizes the bias to the format of the accumulator (INT32, or 62 bits for block-scaled tensors), with the scale of the accumulator: input scale × weight scale, for each output channel. Returns the fake-quantized bias (for the training mode) and its int64 codes (for the integer mode). The bias gets a straight-through gradient; the scales get none.

| Argument | Type | Meaning |
|---|---|---|
| `bias` | `torch.Tensor` | The float bias, one for each output channel |
| `x_params` | `QuantParams` | The parameters of the input |
| `w_params` | `QuantParams` | The parameters of the weight |
| `mode` | `RoundMode` | The rounding rule |

```python
x_params = QuantParams(torch.tensor(0.1), torch.tensor(0.0), build_format("int8"))
w_params = QuantParams(torch.tensor([0.01, 0.02]), torch.zeros(2), build_format("int8"))
fake, codes = bias_quantize(torch.tensor([0.1234, -0.05]), x_params, w_params, "half_away")
print(fake, codes)  # tensor([ 0.1230, -0.0500]) tensor([123, -25])   0.1234 / (0.1 × 0.01) = 123.4
```
