---
title: "types.py"
description: "RequantizeStyle, FixedPoint, OpSet, and the names of activations and NEQ scopes."
sidebar_position: 13
---

The types of the quantized ops.

```python
from quaver.ops import ActivationKind, FixedPoint, NeqScope, OpSet, RequantizeStyle
```

---

## RequantizeStyle

```python
RequantizeStyle(name, multiplier_max, shift_max, round_mode)
```

Describes how a device rescales an accumulator: the largest multiplier, the largest shift, and the rounding. `REQUANTIZE_REGISTRY` holds the styles by name. `multiplier_max = 0` means "exact": the real ratio is kept in float64, with no multiplier and shift. It is a frozen dataclass.

| Argument | Type | Meaning |
|---|---|---|
| `name` | `str` | The name of the style |
| `multiplier_max` | `int` | The largest multiplier (65535 for a 16-bit multiplier, 1 for a shift only, 0 for exact) |
| `shift_max` | `int` | The largest right shift |
| `round_mode` | `RoundMode` | The rounding of the final shift |

```python
style = RequantizeStyle("mul16_shift", multiplier_max=2**16 - 1, shift_max=94, round_mode="half_up")
```

---

## FixedPoint

```python
FixedPoint(multiplier, shift, style)
```

Holds the multiplier and the shift that stand for a ratio of scales: ratio ≈ multiplier / $2^{\text{shift}}$. `requantize_params()` and `requantize_ratio()` return it.

| Argument | Type | Meaning |
|---|---|---|
| `multiplier` | `torch.Tensor` | The integer multiplier $M_0$ (the ratio itself, in float64, for the exact style) |
| `shift` | `torch.Tensor` | The shift $n$ (negative for a left shift) |
| `style` | `RequantizeStyle` | The style they were made for |

```python
fixed = requantize_params(torch.tensor(0.002, dtype=torch.float64), "mul16_shift")
print(fixed.multiplier, fixed.shift)  # tensor(33554) tensor(24)
```

---

## OpSet

```python
OpSet(requantize, add, pool_avg2d, matmul, normalize_l1, normalize_l1_scores, activation, block_quantize)
```

Holds one version (training, integer, or float) of every op, so that a layer writes its forward pass only once. `mode_ops()` returns the set for the current mode. The versions of an op take the same arguments.

| Argument | Type | Meaning |
|---|---|---|
| `requantize` | `Callable` | `requantize_fake`, `requantize_integer`, or `requantize_float` |
| `add` | `Callable` | The residual add |
| `pool_avg2d` | `Callable` | The average pooling |
| `matmul` | `Callable` | The matrix product |
| `normalize_l1` | `Callable` | NEQ |
| `normalize_l1_scores` | `Callable` | The attention map of SimA |
| `activation` | `Callable` | ReLU and ReLU6 |
| `block_quantize` | `Callable` | The quantization to block scales |

```python
ops = mode_ops(integer_mode=False)
print(ops.requantize.__name__)  # requantize_fake
```

---

## ActivationKind, NeqScope

```python
ActivationKind = Literal["none", "relu", "relu6"]
NeqScope       = Literal["token", "tensor"]
```

The names of the activations of the layers, and of the scope of NEQ in SimA (one scale for each token, or one for each sample and head).

| Name | Values |
|---|---|
| `ActivationKind` | `"none"`, `"relu"`, `"relu6"` |
| `NeqScope` | `"tensor"` (the paper's deployed setting), `"token"` |

```python
conv = QuantConv2d(16, 32, 3, padding=1, output_format="uint8", activation="relu6")
```
