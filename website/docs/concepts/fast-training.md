---
title: "Fast training"
description: "Why QAT with exact integer math stays fast: no syncs, CUDA graphs, and float32 math."
---

Quaver repeats the integer math of your device in every training step, so a step has many small GPU tasks. To keep QAT fast anyway, Quaver never makes the CPU wait for the GPU. So you can capture the whole training step as one CUDA graph, and you can run the math in float32.

## The time of one epoch

ResNet-8 on CIFAR-10, with a batch size of 128, on an RTX 3060. The float model with no quantization takes 2.7 s.

| How the step runs | float64 math (default) | float32 math |
|---|---|---|
| As it is | 21.0 to 23.1 s | 13.0 to 14.1 s |
| As a CUDA graph | 18.0 to 20.0 s | 10.9 to 11.8 s |

## No syncs in a training step

A sync is a moment when the CPU stops and waits for a value from the GPU, for example with `.item()` or `if (x < 0).any():`. Quaver never does this in a training step. The scales, the multipliers and shifts, and even the stop of the range updates are all computed on the GPU.

So the CPU can queue up work ahead of the GPU. And a training step can be captured as a CUDA graph, which needs a step with no syncs.

When you write your own layer, check it with `sync_count()`. It runs a function once, and returns the number of syncs.

```python
from quaver.model_tools import sync_count

print(sync_count(layer, x, x_params))  # 0
```

## Capture the step as a CUDA graph

A CUDA graph records all the GPU tasks of one step once, and replays them with one call. This saves the time to launch thousands of small tasks. The examples capture the step like this:

```python
# Fixed tensors: a captured step always reads the same memory
images = torch.zeros((128, 3, 32, 32), device="cuda")
labels = torch.zeros(128, dtype=torch.int64, device="cuda")

# The learning rate lives in a tensor, so that the scheduler can change it after the capture
optimizer = torch.optim.AdamW(model.parameters(), lr=torch.tensor(1e-4, device="cuda"), capturable=True)

# train_step() is your usual step: forward, loss, zero_grad(set_to_none=True), backward, and optimizer.step()
# Warm up for a few steps on a side stream, and then capture one step
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    for _ in range(3):
        train_step(model, optimizer, images, labels)
torch.cuda.current_stream().wait_stream(side)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    train_step(model, optimizer, images, labels)

# For each batch: copy it into the fixed tensors, and replay the step
for batch_images, batch_labels in loader:
    images.copy_(batch_images)
    labels.copy_(batch_labels)
    graph.replay()
```

Every batch must have the same shape, so leave out the last batch if it is smaller. Capture the graph again after you switch a mode or change the math type, because a graph keeps what it recorded.

## Train in float32

By default, the training mode computes in float64, so that its codes match the integer mode exactly. You can run the training steps in float32 instead. It is about 1.6 times faster, because the GPUs for games run float64 very slowly.

```python
from quaver import precision_set_math

precision_set_math(train="float32")  # tests (without gradients) stay in float64
```

In float32, a code can be one off in rare places during training. Tests still run in float64, so they match the integer mode exactly. Quaver tells a test from a training step by the gradients, not by `eval()`. So run your tests inside `torch.no_grad()` (or `torch.inference_mode()`). With gradients on, a test also runs in float32. In a test with 15 epochs of QAT, the accuracy did not drop:

| Formats | float64 | float32 |
|---|---|---|
| INT8 / UINT8 | 82.39% | 82.41% |
| INT4 / UINT4 | 77.71% | 78.18% |
| FP4 / UFP4 | 81.17% | 81.04% |
| FP4 / FP4 | 76.35% | 76.74% |

## Stop the range updates on the GPU

The scales of the activations usually stop changing after a few epochs. `freeze_scalers(model, step=...)` makes them stop by themselves after that many steps. The count lives on the GPU, so the stop needs no sync, and a captured graph does not need to be captured again.

```python
from quaver import freeze_scalers

freeze_scalers(model, step=5 * steps_per_epoch)  # stop after 5 epochs
```

## Block-scaled layers are slower

Block-scaled layers (NVFP4 and others) also run their integer math in the training mode, because their sums are too wide for float. So they train more slowly than the other layers.
