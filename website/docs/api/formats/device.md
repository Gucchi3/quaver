---
title: "device.py"
description: "Register the integer steps of the device as torch ops (torch.ops.quaver.*), so that the export keeps each step as one node."
sidebar_position: 8
---

Registers the integer steps of the device as torch ops, under `torch.ops.quaver`. `model_export()` captures the model with `torch.export`, which then keeps each of these steps as one node, such as `quaver.requantize`. The steps themselves are written next to their ops, as functions named `*_device`.

```python
from quaver.formats import DEVICE_LIBRARY, device_define
```

---

## DEVICE_LIBRARY

```python
DEVICE_LIBRARY: torch.library.Library
```

The torch library `quaver`, which holds every device op. Each op runs the same Python code on every device, and on the fake tensors of `torch.export`.

```python
print(torch.ops.quaver.quantize)  # quaver.quantize
```

---

## device_define

```python
device_define(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]
```

Returns a decorator that registers a function as the op `torch.ops.quaver.<name>`, and gives the function back unchanged. The schema of the op comes from the type hints of the function, so the function must take plain tensors, ints, floats, and strings (no `QuantParams`). Call the op through `torch.ops.quaver.<name>`: then `torch.export` sees one node. Calling the function directly also works, but `torch.export` then sees the steps inside it.

| Argument | Type | Meaning |
|---|---|---|
| `name` | `str` | The name of the op under `torch.ops.quaver` |

```python
from quaver.formats import device_define

@device_define("double_codes")
def double_codes_device(codes: torch.Tensor) -> torch.Tensor:
    return codes * 2

print(torch.ops.quaver.double_codes(torch.tensor([1, 2])))  # tensor([2, 4])
```
