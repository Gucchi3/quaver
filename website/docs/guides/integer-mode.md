---
title: "Check with the integer mode"
description: "Run the trained model with integer math only, and check that it gives the same logits as the training mode."
---

After training, run the model in the [integer mode](../concepts/modes.md). Each layer then takes integer codes and computes with integer math only, as your device will. If the logits are the same as in the training mode, the accuracy you measured is the accuracy of your device.

## Compare the two modes on the test set

```python
from quaver import params_values, switch_integer_mode

model.eval()                 # fix the scales
same    = True
correct = 0
with torch.no_grad():        # tests run in float64, so they match exactly (see Fast training)
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Training mode
        switch_integer_mode(model, False)
        fake = model(images)

        # Integer mode
        switch_integer_mode(model, True)
        integer = params_values(*model(images))   # the model returns (codes, QuantParams) in the integer mode

        same    &= torch.equal(fake, integer)
        correct += (integer.argmax(dim=1) == labels).sum().item()

print("same logits:", same)                                   # True
print("accuracy:", correct / len(test_loader.dataset))
```

`torch.equal()` checks that every logit of every image is exactly the same, not just close. If it prints `False`, check the list in [How the modes match bit for bit](../concepts/bit-exact.md#what-you-need-for-a-match). The usual causes are TF32 (call `precision_disable_tf32()` at the start), a test without `torch.no_grad()` after `precision_set_math(train="float32")`, and a step in your own layer that rounds without `mode_ops()`.

## Look at the codes of one layer

To check your device layer by layer, you need the codes that each layer gives. A forward hook can keep them:

```python
codes = {}

def keep_codes(name):
    def hook(module, inputs, output):
        codes[name] = output[0]    # the codes (int64); output[1] is their QuantParams
    return hook

for name, module in model.named_children():
    if name != "quantizer":
        module.register_forward_hook(keep_codes(name))

switch_integer_mode(model, True)
with torch.no_grad():
    model(images[:1])

print(codes["conv1"].shape, codes["conv1"].dtype)  # torch.Size([1, 32, 32, 32]) torch.int64
```

These codes are what your device must give for the same input image. Save them with the input codes, and use them as test vectors. [Device math](../api/device-math.md) lists the steps your device runs to get them.

Next, you export the model for your device.
