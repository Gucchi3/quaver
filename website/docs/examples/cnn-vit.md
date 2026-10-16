---
title: "CNN + ViT"
description: "A small vision transformer on CIFAR-10 with SimA attention and residual scale sharing, all in integer math."
---

This example adds transformer blocks to a small CNN. Its attention is SimA, which needs no softmax, so the whole model runs with integer math only, attention included. The residual stream of each block shares one scale, so the adds need no rescaling on a device.

## Run it

Run it from the top of the repository. It needs a GPU, and downloads CIFAR-10 into `example/cnn_vit/data/` the first time.

```bash
python -m example.cnn_vit.main                         # FP32, then all five quantized models
python -m example.cnn_vit.main --models int8 int4
python -m example.cnn_vit.main --cuda-graph --float32  # faster
```

As in the [ResNet-8 example](resnet8.md), it trains the float model for 40 epochs, folds the BatchNorms, and runs 15 epochs of QAT for each quantized model. The results go to `example/cnn_vit/log/`.

## Results

| Model | Weights | Activations | Training mode | Integer mode | Modes match |
|---|---|---|---|---|---|
| fp32 | fp32 | fp32 | 83.09% | | |
| int8 | int8 | uint8 | 83.15% | 83.15% | yes |
| int8_asym | int8 (zero point) | int8 (zero point) | 83.04% | 83.04% | yes |
| int4 | int4 | uint4 | 79.47% | 79.47% | yes |
| int4_asym | int4 (zero point) | int4 (zero point) | 79.90% | 79.90% | yes |
| fp4 | fp4 | ufp4 | 81.86% | 81.86% | yes |

## The model

```text
images (uint8, scale 1/255)
→ conv1: QuantConv2d 3×3, 16 channels, ReLU
→ conv2: QuantConv2d 3×3, 32 channels, stride 2, ReLU
→ embed: QuantConv2d 3×3, 64 channels, stride 2 (no ReLU) → 8 × 8 = 64 tokens of 64 channels
→ 2 × QuantSimaBlock (4 heads, key and value size 16, MLP 128)
→ classifier: QuantLinear 64 × 64 → 10 on all the tokens (INT32 logits)
```

The tokens stay in the shape of an image (B, C, H, W), so the attention and the MLP use 1×1 convs as their linear layers. There is no pooling: the classifier reads every token.

### Normalization

Transformers usually use LayerNorm. This model puts a BatchNorm after each conv instead, like LeViT. Before QAT, `fold_batch_norms()` folds every BatchNorm into the weights of its conv, so the quantized model has no normalization layer at all, and the device has less to do.

### One block

```text
x = the block input, requantized to the shared scale of the block
x = x + attention(x)                      (INT8 add)
x = x + conv1×1(ReLU(conv1×1(x)))         (INT8 add, the MLP)
```


### SimA attention

`QuantSimaAttention` makes Q, K, and V with 1×1 convs. In place of softmax, SimA normalizes each token of Q and K by its L1 norm. Quaver does this with integer math: the device finds the L1 norm and the scale with integer division, and requantizes in the same step. Then it computes Q Kᵀ, multiplies by V, and projects the result with a 1×1 conv. [SimA attention](../learn/sima-attention.md) explains the math.

Because nothing in it needs exp or a float division, the attention runs on the same integer hardware as the convs.

### Residual scale sharing

In a normal residual add, the two inputs have different scales, so the device must rescale both before it adds them. Here, one `SharedScaler` gives one scale to the whole residual stream of a block: the block input, the outputs of the attention and the MLP, and both sums. The branches requantize their outputs to that scale (in the same step as their last conv), so each add is a plain INT8 add with saturation ([Shared scales](../learn/shared-scales.md)). The next block starts its own scale.

### The 4-bit models

In `int4`, `int4_asym`, and `fp4`, everything but the first conv, the classifier, and the residual stream is 4 bits. The residual stays INT8, and is requantized to 4 bits at the input of the attention and of the MLP. SimA keeps symmetric quantizers in every model, and the input of its projection is signed, since there is no ReLU after attention × V.
