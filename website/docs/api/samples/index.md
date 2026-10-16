---
title: "samples/"
description: "Sample layers: Conv2d, Linear, block-scaled Conv2d and Linear, SimA attention, residual add, and average pooling."
sidebar_position: 1
---

Ready-made quantized layers. Each one takes and returns a pair of a tensor and its `QuantParams`, and runs in all three modes. Use them as they are, or as models for your own layers.

| File | What it holds |
|---|---|
| [quant_conv2d.py](quant_conv2d.md) | `QuantConv2d` |
| [quant_linear.py](quant_linear.md) | `QuantLinear` |
| [quant_layer.py](quant_layer.md) | `QuantLayer`: the shared forward pass, and the base class for your own conv-like layers |
| [quant_add.py](quant_add.md) | `QuantAdd`: the residual add |
| [quant_avg_pool2d.py](quant_avg_pool2d.md) | `QuantAvgPool2d` |
| [quant_block_conv2d.py](quant_block_conv2d.md) | `QuantBlockConv2d`: NVFP4 and other block-scaled convs |
| [quant_block_linear.py](quant_block_linear.md) | `QuantBlockLinear` |
| [quant_block_layer.py](quant_block_layer.md) | `QuantBlockLayer`: the shared forward pass of the block-scaled layers |
| [quant_sima_attention.py](quant_sima_attention.md) | `QuantSimaAttention`: SimA attention without softmax |

It uses formats/, scales/, quantizers/, and ops/.
