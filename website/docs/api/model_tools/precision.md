---
title: "precision.py"
description: "Float settings: turn off TF32, and choose the float type of the training-mode math."
sidebar_position: 8
---

Float settings that matter for exact integer results and for speed. [Fast training](../../concepts/fast-training.md#train-in-float32) explains float32 math.

```python
from quaver import precision_disable_tf32, precision_set_math
from quaver.model_tools import MATH_TYPES
```

---

## precision_disable_tf32

```python
precision_disable_tf32() -> None
```

Turns off TF32 in the convolutions and the matrix products of PyTorch, for the whole process. TF32 keeps only 10 bits of the mantissa, so the training mode could not get the exact INT32 sums back. Call it once, at the start of your program. It made training about 10% slower on an RTX 3060, and changes nothing on a CPU.

```python
precision_disable_tf32()
```

---

## MATH_TYPES

```python
MATH_TYPES: dict[str, torch.dtype]
```

The names that `precision_set_math()` takes, and their float types.

```python
print(MATH_TYPES)  # {'float32': torch.float32, 'float64': torch.float64}
```

---

## precision_set_math

```python
precision_set_math(train: MathDtype = "float64", test: MathDtype = "float64") -> None
```

Chooses the float type of the training-mode math: `train` while gradients are on (training steps), and `test` while they are off (inside `torch.no_grad()`). float32 makes training about 1.6 times faster; then a code can be one off the device in rare places during training, but tests in float64 stay exact. The integer mode always uses float64. Call it before you capture a CUDA graph.

| Argument | Type | Meaning |
|---|---|---|
| `train` | `MathDtype` | `"float64"` or `"float32"`, for training steps |
| `test` | `MathDtype` | `"float64"` or `"float32"`, for tests |

```python
precision_set_math(train="float32")   # faster training; tests stay float64
```
