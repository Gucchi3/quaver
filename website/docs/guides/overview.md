---
title: "The code you will build"
description: "The whole program of the guides: a small quantized CNN for CIFAR-10, trained with QAT and checked with the integer mode."
---

In these guides, you build one program, step by step. It makes a small quantized CNN for CIFAR-10, trains it with QAT, checks that the integer mode gives the same logits, and exports it for your device. Here is the whole program. Each guide explains one part of it.

| Part of the code | Guide |
|---|---|
| The quantizer of the input | [Write a quantizer](quantizer.md) |
| The pairs of values and QuantParams that flow through the model | [Quantized values and QuantParams](quant-params.md) |
| `TinyNet` and its sample layers | [Use the sample layers](sample-layers.md) |
| What each layer does inside | [Inside a quantized layer](inside-a-layer.md) |
| `QuantDepthwiseConv2d`, a layer of your own | [Write your own layer](custom-layer.md) |
| Loading the data, calibration, and the training loop | [Train the model](training.md) |
| The check with the integer mode | [Check with the integer mode](integer-mode.md) |
| Export for your device | [Export to your device](export.md) |

Save it as `tiny_net.py` in your project, next to the `quaver/` folder, and run it with `python tiny_net.py`. It downloads CIFAR-10 (about 170 MB) the first time. It needs torchvision for the data, and runs on a GPU or a CPU (a GPU is much faster).

```python
import itertools
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from torch import nn

from quaver         import Granularity, QuantAvgPool2d, QuantConv2d, QuantLinear, Quantizer, RangeScaler, build_format, calibrate_model, model_export, params_values, precision_disable_tf32, switch_integer_mode
from quaver.ops     import accumulate_conv2d
from quaver.samples import QuantLayer


# A layer of your own: a depthwise conv (Write your own layer)
class QuantDepthwiseConv2d(QuantLayer):
    output_axis = 1

    def __init__(self, channels, kernel_size, stride=1, padding=0, output_format=None, activation="none"):
        conv = nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels)
        super().__init__(conv, "int8", output_format, activation, "mul16_shift", None, True, True, None)
        self.stride  = stride
        self.padding = padding
        self.groups  = channels

    def _layer_float(self, x, weight, bias):
        return F.conv2d(x, weight, bias, self.stride, self.padding, groups=self.groups)

    def _layer_integer(self, q_x, x_params, q_w, w_params):
        return accumulate_conv2d(q_x, x_params, q_w, w_params, self.stride, self.padding, 1, self.groups)


# The model (Use the sample layers)
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantizer    = Quantizer(build_format("int8"), RangeScaler(), Granularity(), batch_axis=0)
        self.conv1        = QuantConv2d(3, 32, 3, padding=1, output_format="uint8", activation="relu")
        self.depthwise2   = QuantDepthwiseConv2d(32, 3, stride=2, padding=1, output_format="uint8", activation="relu")
        self.pointwise2   = QuantConv2d(32, 64, 1, output_format="uint8", activation="relu")
        self.depthwise3   = QuantDepthwiseConv2d(64, 3, stride=2, padding=1, output_format="uint8", activation="relu")
        self.pointwise3   = QuantConv2d(64, 128, 1, output_format="uint8", activation="relu")
        self.pool         = QuantAvgPool2d(8)
        self.classifier   = QuantLinear(128, 10)
        self.integer_mode = False

    def forward(self, images):
        x, params = self.quantizer.encode(images) if self.integer_mode else self.quantizer(images)
        x, params = self.conv1(x, params)
        x, params = self.pointwise2(*self.depthwise2(x, params))
        x, params = self.pointwise3(*self.depthwise3(x, params))
        x, params = self.pool(x, params)
        x, params = self.classifier(x.flatten(1), params)
        return (x, params) if self.integer_mode else x


def main():
    # 1. Load CIFAR-10 (Train the model)
    precision_disable_tf32()
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    normalize    = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_data   = torchvision.datasets.CIFAR10("data", train=True , download=True, transform=T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), normalize]))
    test_data    = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=T.Compose([T.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    test_loader  = torch.utils.data.DataLoader(test_data , batch_size=500)

    # 2. Make the model, and calibrate its scales (Train the model)
    model = TinyNet().to(device)
    calibrate_model(model, (images.to(device) for images, _ in itertools.islice(train_loader, 20)))

    # 3. Train with QAT (Train the model)
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

    # 4. Check that the integer mode gives the same logits (Check with the integer mode)
    model.eval()
    same    = True
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            switch_integer_mode(model, False)
            fake = model(images)
            switch_integer_mode(model, True)
            integer = params_values(*model(images))
            same    &= torch.equal(fake, integer)
            correct += (integer.argmax(dim=1) == labels).sum().item()
    print("same logits:", same)
    print("accuracy:", correct / len(test_data))

    # 5. Export model.json and model.bin for your device (Export to your device)
    switch_integer_mode(model, True)
    model_export(model.cpu(), (images[:1].cpu(),), "tiny_net_export")


# The data loader starts worker processes, so the program runs from here (needed on Windows)
if __name__ == "__main__":
    main()
```

It prints the loss of each epoch, and at the end:

```text
same logits: True
accuracy: 0.59...
```

The accuracy is about 60%, because the model is very small. "same logits: True" means that the integer mode, which runs the same math as your device, gives exactly the same logits as the model you trained. At the end, the program writes `tiny_net_export/model.json` and `tiny_net_export/model.bin` for your device.
