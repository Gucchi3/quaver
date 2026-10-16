---
title: "quant_block_linear.py"
description: "QuantBlockLinear: Linear with block-scaled weights (NVFP4 and the like), with integer math only."
sidebar_position: 7
---

The block-scaled Linear. With fp4 weights and ue4m3 block scales, it is NVFP4. With INT8 inputs, it is W4A8.

```python
from quaver import QuantBlockLinear
```

---

## QuantBlockLinear

```python
QuantBlockLinear(in_features, out_features, bias=True, weight_format="fp4", scale_format="ue4m3", input_format=None, output_format="int8", output_block=False, activation="none", style="mul16_shift", block_size=16, output_scaler=None)
```

Makes a Linear with block-scaled weights (a scale for each block of 16 input features), optional block-scaled inputs, an exact wide accumulator, and the output in the format of the next op. `in_features` must be a multiple of `block_size`.

| Argument | Type | Meaning |
|---|---|---|
| `in_features` | `int` | The input features |
| `out_features` | `int` | The output features |
| `bias` | `bool` | Whether the layer has a bias (stored with the wide accumulator) |
| `weight_format` | `str` | The format of the weight elements, such as `"fp4"` |
| `scale_format` | `str` | The format of the block scales: `"ue4m3"` for NVFP4, or `"uint6"` / `"uint10"`. Use `"uint6"` for integer devices: only its sums fit in 32 bits ([why](../../guides/block-scale.md#the-width-of-the-accumulator)) |
| `input_format` | `str \| None` | Quantize the input to blocks in this format here (such as `"ufp4"` after a ReLU), or `None` to take the input as it is (block-scaled, or INT8) |
| `output_format` | `str` | The format of the output |
| `output_block` | `bool` | `True` gives a block-scaled output, for the next block-scaled layer; `False` requantizes to one fixed scale |
| `activation` | `ActivationKind` | `"none"`, `"relu"`, or `"relu6"` |
| `style` | `str` | How the device requantizes |
| `block_size` | `int` | The number of input channels in a block |
| `output_scaler` | `Scaler \| None` | A scaler for the output (when `output_block` is `False`), such as a `SharedScaler` |

```python
linear = QuantBlockLinear(32, 10, output_format="int8")   # NVFP4 weights, INT8 inputs (W4A8)
```

### forward

```python
linear(x, x_params) -> tuple[torch.Tensor, QuantParams]
```

Runs the layer in the current mode on the last axis of `x`, and returns the output and its parameters.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The input: fake-quantized values in the training mode, codes in the integer mode, float values in the float mode |
| `x_params` | `QuantParams` | The parameters of the input (one scale for the tensor) |

```python
quantizer = Quantizer(build_format("int8"), RangeScaler(), batch_axis=0)
x         = torch.randn(4, 32)
linear(*quantizer(x))
quantizer.eval(); linear.eval()
y, y_params = linear(*quantizer(x))
print(y.shape, y_params.number_format.name)  # torch.Size([4, 10]) int8
```
