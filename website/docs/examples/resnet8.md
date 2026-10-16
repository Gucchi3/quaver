---
title: "ResNet-8"
description: "QAT of ResNet-8 on CIFAR-10 in six formats, from INT8 to NVFP4, with every layer in integer math."
---

This example trains ResNet-8 on CIFAR-10 in float, and then fine-tunes six quantized versions of it with QAT. In every version, all the layers run with integer math in the integer mode, and the integer mode gives the same logits as the training mode on all 10,000 test images.

## Run it

Run it from the top of the repository. It needs a GPU, and downloads CIFAR-10 into `example/resnet8/data/` the first time.

```bash
python -m example.resnet8.main                             # FP32, then all six quantized models
python -m example.resnet8.main --models int8 nvfp4         # only some of them
python -m example.resnet8.main --cuda-graph --float32      # faster (see Fast training)
python -m example.resnet8.main --fp32-epochs 1 --qat-epochs 1   # a quick try
```

It trains the float model for 40 epochs, folds its BatchNorms into the convs, and loads the weights into each quantized model. Then it runs 15 epochs of QAT for each one. The activation ranges stop changing after 3 epochs. The weights and a table of the results go to `example/resnet8/log/`.

## Results

| Model | Weights | Activations | Training mode | Integer mode | Modes match |
|---|---|---|---|---|---|
| fp32 | fp32 | fp32 | 81.86% | | |
| int8 | int8 | uint8 | 82.00% | 82.00% | yes |
| int8_asym | int8 (zero point) | int8 (zero point) | 82.04% | 82.04% | yes |
| int4 | int4 | uint4 | 78.51% | 78.51% | yes |
| int4_asym | int4 (zero point) | int4 (zero point) | 78.59% | 78.59% | yes |
| fp4 | fp4 | ufp4 | 80.50% | 80.50% | yes |
| nvfp4 | fp4 + ue4m3 per 16 | ufp4 + ue4m3 per 16 | 81.71% | 81.71% | yes |

"Modes match" means that every logit of every test image is the same in the two modes. INT8 keeps the accuracy of the float model. NVFP4 loses almost nothing, even with 4-bit weights and activations.

## The model

```text
images (uint8, scale 1/255)
→ stem: QuantConv2d 3×3, 16 channels, ReLU
→ block 1: QuantBasicBlock, 16 channels
→ block 2: QuantBasicBlock, 32 channels, stride 2
→ block 3: QuantBasicBlock, 64 channels, stride 2
→ QuantAvgPool2d 8×8
→ classifier: QuantLinear 64 → 10 (INT32 logits)
```

### The input needs no quantizer

The images are already uint8, so the model uses the pixels as codes, with a scale of 1/255. In the integer mode, the first layer gets the raw pixels. This is also what a device gets from a camera.

### The stem and the classifier are INT8

The first and the last layers are the most sensitive to rounding, so they keep INT8 weights in every model. The classifier returns its INT32 sums as the logits, with no requantization.

### One block

```text
x ──┬── conv1 3×3 (ReLU) → conv2 3×3 (INT32 sums) ──┐
    │                                               QuantAdd (INT8) → ReLU → requantize → out
    └── shortcut: x as it is, or a 1×1 conv (INT8) ─┘
```

- `conv1` ends in the activation format of the model (uint8, int4, ufp4, …).
- `conv2` has no `output_format`, so it returns its INT32 sums. The add requantizes them, so the branch is rounded only once.
- The shortcut is the input as it is, or a 1×1 conv into INT8 when the shape changes.
- `QuantAdd` adds the two inputs into INT8. It brings both to one scale with a multiplier and a shift, adds them, and rounds once ([Residual add](../learn/residual-add.md)).
- After the add, ReLU runs on the codes, and the block output is requantized to the activation format.

The block is a class of the example, `QuantBasicBlock`, written from the sample layers and one op (`ops.activation()`), in the same way as in [Write your own layer](../guides/custom-layer.md).

### Symmetric and asymmetric models

The symmetric models keep every zero point at 0. Their activations come after a ReLU and are never negative, so they use unsigned formats: uint8, uint4, ufp4. The asymmetric models (`_asym`) give every weight and activation a [zero point](../learn/asymmetric.md), so they use signed formats, and 0 sits at the lowest code. FP formats are always symmetric.

### The NVFP4 model

`nvfp4` uses `QuantBlockConv2d` for the convs in the blocks. The weights are fp4, and the inputs of the convs are ufp4, each with a ue4m3 scale for every 16 values ([Block scales](../learn/block-scales.md)). The stem, the classifier, and the residuals stay INT8 or UINT8. The block-scaled convs also run with integer math only: the block codes are integers, and the sums are exact up to 62 bits.

## Add your own setting

The models come from `MODEL_REGISTRY` in `example/resnet8/model/build.py`. To try another mix, add a line:

```python
"int4_int8": Resnet8Setting(weight_format="int4", act_format="uint8"),
```

and run `python -m example.resnet8.main --models int4_int8`.
