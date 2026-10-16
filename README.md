<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/quaver-logo-dark.svg">
    <img src="docs/img/quaver-logo-light.svg" alt="Quaver" width="440">
  </picture>
</p>

A quantization library for embedded devices. Quaver quantizes AI models trained in PyTorch, and exports the data you need to run them on your device.

For details, see the [documentation](https://gucchi3.github.io/quaver/).

## Features

### Many formats and shapes

Quantize image and language models in many formats, from INT8 down to FP4 and NVFP4. Every layer can run with integer math only, even recent ones such as SimA attention.

### Built for embedded devices

Quaver's integer mode matches your device bit for bit, so you can check your hardware with its outputs. You can also export every number your device needs, packed for its memory.

### Fast QAT

QAT stays fast, even with exact integer math. The CPU never waits for the GPU, so you can run each training step as one CUDA graph.

### Learn quantization

The docs explain quantization, from the basics to the integer math on your device.

## Quick start

Quaver needs Python 3.14 or later and PyTorch 2.14 or later. Clone this repository, and copy the `quaver/` folder inside it into your project. If you put it in a subfolder such as `utils/`, import it as `utils.quaver`.

Each Quaver layer has a training mode and an integer mode. The training mode uses fake-quantized values for QAT. The integer mode uses integer codes, like your device. This code checks that the two modes give the same integer codes.

```python
import torch

from quaver import Granularity, QuantConv2d, Quantizer, RangeScaler, build_format, params_codes, precision_disable_tf32, switch_integer_mode

# 1. Turn off TF32, so that the two modes match exactly on a GPU
precision_disable_tf32()

# 2. Make an INT8 quantizer for the input, and a conv layer with an INT8 output
quantizer = Quantizer(build_format("int8"), RangeScaler(), Granularity(), batch_axis=0)
conv      = QuantConv2d(3, 16, 3, padding=1, output_format="int8")
images    = torch.randn(8, 3, 32, 32)

# 3. Training mode: measure the value ranges in one pass, and then freeze them with eval()
conv(*quantizer(images))
quantizer.eval()
conv.eval()
y, y_params = conv(*quantizer(images))

# 4. Integer mode: give integer codes, and get the same output codes as the training mode
switch_integer_mode(conv, True)
q_x, x_params = quantizer.encode(images)
q_y, _        = conv(q_x, x_params)
print(torch.equal(q_y, params_codes(y, y_params)))  # True
```

To quantize your own model, build it from Quaver layers. They keep the weight names of the PyTorch layers, so you can load the weights of your trained float model. The [documentation](https://gucchi3.github.io/quaver/) shows each step, from training to export.

## License

Quaver is released under the [Mozilla Public License 2.0](LICENSE).

If you use Quaver, please leave a note in [GitHub Discussions](https://github.com/Gucchi3/quaver/discussions). The author will be very happy to read it. If you use it in your work, please cite it ([CITATION.cff](CITATION.cff)). If you improve it, please send a pull request.
