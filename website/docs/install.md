---
title: "Install"
description: "What you need, and how to add Quaver to your project."
---

## Requirements

| Software | Version |
|---|---|
| Python | 3.14 or later |
| PyTorch | 2.14 or later |

Quaver needs nothing else but the Python standard library. It runs on a CPU and on a CUDA GPU. The training examples in `example/` use a CUDA GPU.

## Add Quaver to your project

Quaver is not a pip package. You get the repository, and copy one folder from it into your project.

### 1. Get the repository

Clone the repository with git. You can also download it as a ZIP file from the GitHub page (Code, then Download ZIP).

```bash
git clone https://github.com/OWNER/quaver.git
```

### 2. Copy the Quaver folder into your project

The cloned `quaver/` folder has another `quaver/` folder inside it. This inner folder is the library. Copy only this folder to any place in your project.

```text
quaver/                  ← the cloned repository
├── quaver/              ← the library: copy this folder
├── example/
├── website/
└── ...

my_project/            ← your project
├── main.py
└── utils/
    └── quaver/          ← paste it here
```

### 3. Import it

All imports inside the library are relative, so it works from any folder. Write your imports to match the place where you put it. You can also rename the folder.

```python
from utils.quaver import QuantConv2d, Quantizer
```

You can import the common classes and functions from the top of the library. Import everything else from its subpackage, for example `from utils.quaver.ops import mode_ops`. The [API reference](api/index.md) shows where each name is.

## Next step

[Try it in 5 minutes](quickstart.md) runs a quantized layer, and checks that its two modes give the same integers.
