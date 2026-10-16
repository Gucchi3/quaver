---
title: "quant_sima_attention.py"
description: "QuantSimaAttention: SimA attention without softmax, with NEQ, all in integer math."
sidebar_position: 10
---

SimA attention: Q and K are L1-normalized for each token and quantized with NEQ, and then (Q Kᵀ) V is computed, all with integer math. [SimA attention](../../learn/sima-attention.md) explains it, and the [CNN + ViT example](../../examples/cnn-vit.md) uses it.

```python
from quaver import QuantSimaAttention
```

---

## QuantSimaAttention

```python
QuantSimaAttention(channels, heads, key_dim, value_dim, weight_format="int4", act_format="int4", project_format="int4", activation="none", neq_scope="tensor", output_format=None, output_scaler=None, style="mul16_shift", round_mode=None)
```

Makes SimA attention on (B, C, H, W), with the H × W positions as the tokens: four 1×1 convs for Q, K, V, and the projection, NEQ on Q and K, the attention map, and attention × V. By default there is no activation before the projection, as in the paper. NEQ finds its scale for each sample and head (`neq_scope="tensor"`), and the device finds the multipliers of the attention map with integers.

| Argument | Type | Meaning |
|---|---|---|
| `channels` | `int` | The channels of the input and the output |
| `heads` | `int` | The number of heads |
| `key_dim` | `int` | The size of Q and K in each head |
| `value_dim` | `int` | The size of V in each head |
| `weight_format` | `str` | The format of the weights of the four convs |
| `act_format` | `str` | The format of Q, K, the attention map, and V (8 bits or less) |
| `project_format` | `str` | The format of the input of the projection: signed without an activation, unsigned after ReLU or ReLU6 |
| `activation` | `ActivationKind` | `"none"`, `"relu"`, or `"relu6"`, on attention × V |
| `neq_scope` | `NeqScope` | `"tensor"` (one scale for each sample and head) or `"token"` |
| `output_format` | `str \| None` | The format of the output of the projection, or `None` for the INT32 accumulator |
| `output_scaler` | `Scaler \| None` | A scaler for the output, such as the `SharedScaler` of the residual stream |
| `style` | `str` | `"mul16_shift"` or `"shift_only"` (a multiplier and a shift are needed) |
| `round_mode` | `RoundMode \| None` | The rounding rule, or `None` for the usual one |

```python
attention = QuantSimaAttention(64, 4, 16, 16, output_format="int8")
```

### forward

```python
attention(x, x_params) -> tuple[torch.Tensor, QuantParams]
```

Runs SimA attention in the current mode, and returns the output of the projection and its parameters, ready for a residual add. `x_params` must have one scale for the whole tensor.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The input: fake-quantized values in the training mode, codes in the integer mode, float values in the float mode |
| `x_params` | `QuantParams` | The parameters of the input (one scale for the tensor) |

```python
quantizer = Quantizer(build_format("int8"), RangeScaler(), batch_axis=0)
x         = torch.randn(2, 64, 4, 4)   # 16 tokens of 64 channels
attention(*quantizer(x))
quantizer.eval(); attention.eval()
y, y_params = attention(*quantizer(x))
print(y.shape, y_params.number_format.name)  # torch.Size([2, 64, 4, 4]) int8
```
