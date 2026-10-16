---
title: "activation.py"
description: "ReLU and ReLU6 on integer codes: clamps between the code of 0 and the code of 6."
sidebar_position: 7
---

ReLU and ReLU6 on codes. On a device, they are just clamps. Use them on the accumulator, before the requantization, so the limits are exact. [ReLU and ReLU6](../../learn/activations.md) explains them.

```python
from quaver.ops import activation_device, activation_fake, activation_float, activation_integer
```

The examples on this page use an accumulator with the scale 0.5, so the code of 6 is 12:

```python
acc_params = QuantParams(torch.tensor(0.5), torch.tensor(0.0), build_format("int32"))
```

---

## activation_integer

```python
activation_integer(q_x: torch.Tensor, x_params: QuantParams, kind: ActivationKind) -> torch.Tensor
```

The integer mode. Clamps the codes between the code of 0 (the zero point) and, for ReLU6, the code of 6. With `"none"`, the codes stay as they are.

| Argument | Type | Meaning |
|---|---|---|
| `q_x` | `torch.Tensor` | The codes (usually an accumulator) |
| `x_params` | `QuantParams` | Their parameters |
| `kind` | `ActivationKind` | `"none"`, `"relu"`, or `"relu6"` |

```python
print(activation_integer(torch.tensor([-3, 5, 20]), acc_params, "relu"))   # tensor([ 0,  5, 20])
print(activation_integer(torch.tensor([-3, 5, 20]), acc_params, "relu6"))  # tensor([ 0,  5, 12])
```

---

## activation_fake

```python
activation_fake(x: torch.Tensor, x_params: QuantParams, kind: ActivationKind) -> torch.Tensor
```

The training mode. Clamps the codes in the same way as `activation_integer()`, and returns the real values, with straight-through gradients (0 where a value was clamped).

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | Fake-quantized values |
| `x_params` | `QuantParams` | Their parameters |
| `kind` | `ActivationKind` | `"none"`, `"relu"`, or `"relu6"` |

```python
x = torch.tensor([-1.5, 2.5, 10.0], requires_grad=True)
y = activation_fake(x, acc_params, "relu6")
y.sum().backward()
print(y, x.grad)  # tensor([0.0000, 2.5000, 6.0000]) tensor([0., 1., 0.])
```

---

## activation_float

```python
activation_float(x: torch.Tensor, x_params: QuantParams, kind: ActivationKind) -> torch.Tensor
```

The float mode (calibration). ReLU or ReLU6 on the float values.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | Float values |
| `x_params` | `QuantParams` | Not used |
| `kind` | `ActivationKind` | `"none"`, `"relu"`, or `"relu6"` |

```python
print(activation_float(torch.tensor([-1.0, 7.0]), acc_params, "relu6"))  # tensor([0., 6.])
```

---

## activation_device

```python
activation_device(q_x: torch.Tensor, zero_point: torch.Tensor, six: torch.Tensor, kind: ActivationKind) -> torch.Tensor
```

The device op `activation` (`torch.ops.quaver.activation`): ReLU clamps the codes at the zero point, and ReLU6 also at `six`, the code of 6, which is fixed ahead of time.

| Argument | Type | Meaning |
|---|---|---|
| `q_x` | `torch.Tensor` | The codes |
| `zero_point` | `torch.Tensor` | The code of 0 |
| `six` | `torch.Tensor` | The code of 6 (not used by ReLU) |
| `kind` | `ActivationKind` | `"relu"` or `"relu6"` |

```python
print(torch.ops.quaver.activation(torch.tensor([-3, 5, 20]), torch.tensor(0), torch.tensor(12), "relu6"))  # tensor([ 0,  5, 12])
```
