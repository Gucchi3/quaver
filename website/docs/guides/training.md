---
title: "Train the model"
description: "Calibrate the scales, and train the CIFAR-10 model with QAT in a normal PyTorch loop."
---

A Quaver model trains in a normal PyTorch loop. The only extra steps are to turn off TF32 at the start, and to calibrate the scales before the first step.

## Load CIFAR-10

This guide uses torchvision to load the data. Quaver itself does not need it.

```python
import itertools
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from quaver import calibrate_model, precision_disable_tf32

precision_disable_tf32()     # once, at the start (see Try it in 5 minutes)
device = "cuda" if torch.cuda.is_available() else "cpu"

normalize    = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
train_data   = torchvision.datasets.CIFAR10("data", train=True , download=True, transform=T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), normalize]))
test_data    = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=T.Compose([T.ToTensor(), normalize]))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
test_loader  = torch.utils.data.DataLoader(test_data , batch_size=500)
```

## Calibrate the scales

A new model has no scales yet: its range scalers have seen no values. `calibrate_model()` runs a few batches in the [float mode](../concepts/modes.md#float-mode), so that each scaler sees the real range of its values. Then QAT starts from good scales.

```python
model = TinyNet().to(device)
calibrate_model(model, (images.to(device) for images, _ in itertools.islice(train_loader, 20)))
```

## Train

```python
epochs    = 15
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        loss = F.cross_entropy(model(images), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f"epoch {epoch + 1}: loss {loss.item():.3f}")

torch.save(model.state_dict(), "tiny_net.pth")
```

The data loader uses worker processes (`num_workers=2`). On Windows, a program with workers must start from `if __name__ == "__main__":`, as in [the whole program](overview.md).

This small model reaches about 60% test accuracy after 15 epochs. A bigger model, more epochs, or a start from the weights of a trained float model gives more.

In `train()`, the range scalers keep a moving average of the ranges, and the weights learn with the quantization in the loop. The loss sees the same rounding as your device, so the model learns to work with it.

## If you use LSQ

If some quantizers use `LsqScaler`, their scales are parameters that the optimizer trains. Two helpers keep them healthy:

```python
from quaver import lsq_clamp_steps, lsq_param_groups

optimizer = torch.optim.AdamW(lsq_param_groups(model, weight_decay=1e-4), lr=2e-3)  # no weight decay on the LSQ scales

# ... after each optimizer.step():
lsq_clamp_steps(model)  # keep every LSQ scale above a small positive limit
```

## Make it faster

- Stop the range updates after a few epochs, with no sync: `freeze_scalers(model, step=5 * len(train_loader))`.
- Run the training steps in float32: `precision_set_math(train="float32")`.
- Capture the training step as a CUDA graph.

[Fast training](../concepts/fast-training.md) explains each of these.

Next, check the trained model with the integer mode.
