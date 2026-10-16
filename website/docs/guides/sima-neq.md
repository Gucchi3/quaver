---
title: "SimA attention and NEQ"
description: "Build a transformer block with SimA attention, which runs with integer math only, and choose its formats and NEQ scope."
---

`QuantSimaAttention` is an attention that needs no softmax, so a transformer block can run with integer math only. [SimA attention](../learn/sima-attention.md) explains how it works. This page builds a transformer block with it.

## Tokens as an image

`QuantSimaAttention` works on tensors shaped like an image, (B, C, H, W). The H × W positions are the tokens, and C is the size of each token. Q, K, V, and the projection are 1×1 convs. So a CNN can feed its feature map straight into the attention, as in the [CNN + ViT example](../examples/cnn-vit.md).

## A transformer block

```python
from torch import nn

from quaver import QuantAdd, QuantConv2d, QuantSimaAttention


class SimaBlock(nn.Module):
    def __init__(self, channels, heads, key_dim, value_dim, hidden):
        super().__init__()
        self.attention = QuantSimaAttention(channels, heads, key_dim, value_dim, weight_format="int8", act_format="int8", project_format="int8")
        self.add1      = QuantAdd("int8")
        self.expand    = QuantConv2d(channels, hidden, 1, output_format="uint8", activation="relu")
        self.shrink    = QuantConv2d(hidden, channels, 1)
        self.add2      = QuantAdd("int8")

    def forward(self, x, x_params):
        # x + attention(x): the projection gives its INT32 accumulator, and the add requantizes once
        y, y_params = self.attention(x, x_params)
        x, x_params = self.add1(y, y_params, x, x_params)

        # x + MLP(x)
        y, y_params = self.shrink(*self.expand(x, x_params))
        return self.add2(y, y_params, x, x_params)


block = SimaBlock(64, 4, 16, 16, 128)   # 64 channels, 4 heads of size 16, an MLP of 128
```

- The input must have one scale for the whole tensor (INT8 here).
- Without `output_format`, the attention and the last conv of the MLP return their INT32 sums. The adds then requantize them together with the residual, with only one rounding.

## The settings of the attention

| Argument | What to choose |
|---|---|
| `weight_format` | The weights of the four 1×1 convs: `"int8"`, `"int4"` (the default), `"fp4"`, … |
| `act_format` | Q, K, the attention map, and V. 8 bits or less, so the integer math stays exact |
| `project_format` | The input of the projection. Signed (`"int4"`, `"int8"`) without an activation, as in the paper; unsigned after `activation="relu"` or `"relu6"` |
| `activation` | `"none"` (the default) or a ReLU on attention × V |
| `neq_scope` | `"tensor"` (the default): one scale for each sample and head, the setting the paper uses on its device. `"token"`: one scale for each token, a little more exact and more work |
| `output_format`, `output_scaler` | Requantize the projection here, for example to the shared scale of the residual stream ([Scale sharing](scale-sharing.md)) |
| `style` | `"mul16_shift"` or `"shift_only"`: SimA needs a multiplier and a shift |

## What runs on the device

1. Q and K: a 1×1 conv each, then NEQ. The device finds the L1 norm of each token and the scale with an integer division, and quantizes in the same step.
2. The attention map: the integer product Q Kᵀ, requantized with a multiplier that the device finds with integers from the two NEQ scales. $1/\sqrt{d}$ is part of the fixed multiplier.
3. V: a 1×1 conv. Then the integer product of the attention map and V, requantized to `project_format`.
4. The projection: a 1×1 conv.

Nothing in it needs an exponential or a float division. [Device math](../api/device-math.md) lists the exact steps.

## Normalization

Transformers usually use LayerNorm. The CNN + ViT example puts a BatchNorm after each conv instead, like LeViT, and folds it into the conv weights with `fold_batch_norms()` before QAT. Then the quantized block has no normalization layer at all.
