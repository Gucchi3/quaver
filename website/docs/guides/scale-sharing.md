---
title: "Scale sharing"
description: "Give the residual stream of a block one shared scale, so that every residual add is a plain integer add."
---

In a residual block, the adds usually need a rescale of both inputs. With one shared scale for the residual stream, every add becomes a plain integer add with saturation. [Shared scales](../learn/shared-scales.md) explains the idea. This page applies it to the SimA block of [SimA attention and NEQ](sima-neq.md).

## What shares the scale

Inside one block, these tensors share one scale:

- the input of the block (requantized to the shared scale at the start),
- the output of the attention, and the output of the MLP,
- the two sums.

Each branch requantizes its last output to the shared scale anyway, in the same step as its last conv, so this costs nothing extra. The next block starts its own shared scale.

## The block

```python
from torch import nn

from quaver import Granularity, QuantAdd, QuantConv2d, QuantSimaAttention, Quantizer, SharedScaler, build_format
from quaver.ops import mode_ops


class SharedSimaBlock(nn.Module):
    def __init__(self, channels, heads, key_dim, value_dim, hidden):
        super().__init__()
        # One shared scale for the residual stream of this block
        self.shared   = SharedScaler()
        self.residual = Quantizer(build_format("int8"), self.shared, Granularity(), batch_axis=0)

        # The branch outputs and both adds use the shared scale
        self.attention = QuantSimaAttention(channels, heads, key_dim, value_dim, weight_format="int8", act_format="int8", project_format="int8", output_format="int8", output_scaler=self.shared)
        self.add1      = QuantAdd("int8", output_scaler=self.shared)
        self.expand    = QuantConv2d(channels, hidden, 1, output_format="uint8", activation="relu")
        self.shrink    = QuantConv2d(hidden, channels, 1, output_format="int8", output_scaler=self.shared)
        self.add2      = QuantAdd("int8", output_scaler=self.shared)

        self.integer_mode = False
        self.float_mode   = False

    def forward(self, x, x_params):
        ops = mode_ops(self.integer_mode, self.float_mode)

        # Start the pass, and requantize the block input to the shared scale
        self.shared.begin_forward()
        params    = self.residual.params(x) if self.integer_mode else self.residual.measure(x)
        x, params = ops.requantize(x, x_params, params, "mul16_shift"), params

        # x + attention(x), then x + MLP(x): plain INT8 adds on one scale
        y, y_params = self.attention(x, params)
        x, params   = self.add1(y, y_params, x, params)
        y, y_params = self.shrink(*self.expand(x, params))
        x, params   = self.add2(y, y_params, x, params)

        # Move the shared scale toward the widest range of this pass
        self.shared.end_forward()
        return x, params
```

## The rules

1. **Make one `SharedScaler` for the block**, and pass it as `output_scaler` to every layer whose output joins the stream, and to the quantizer of the block input.
2. **Call `begin_forward()` before the first use, and `end_forward()` after the last.** The scale stays the same in between. `end_forward()` moves it toward the widest range seen in the pass.
3. **Give the block the two flags** (`integer_mode`, `float_mode`), since it calls an op itself (`ops.requantize`) to bring its input onto the shared scale.

The shared scale must cover the widest of its tensors, so the smaller ones lose a little resolution. In the [CNN + ViT example](../examples/cnn-vit.md), every block shares its scale this way, and the INT8 model keeps the accuracy of the float model (83.15% against 83.09%).
