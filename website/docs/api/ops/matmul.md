---
title: "matmul.py"
description: "The matrix product of two quantized tensors, for attention."
sidebar_position: 9
---

The matrix product of two quantized tensors, such as Q Kᵀ and the attention map times V. The result is the exact accumulator, with its parameters.

```python
from quaver.ops import matmul_fake, matmul_float, matmul_integer
```

The examples on this page use:

```python
a_params = QuantParams(torch.tensor(0.1) , torch.tensor(0.0), build_format("int8"))
b_params = QuantParams(torch.tensor(0.01), torch.tensor(0.0), build_format("int8"))
```

---

## matmul_integer

```python
matmul_integer(q_a: torch.Tensor, a_params: QuantParams, q_b: torch.Tensor, b_params: QuantParams) -> tuple[torch.Tensor, QuantParams]
```

The integer mode. Returns the exact accumulator (int64) of `a @ b`, and its parameters: the scale is (code scale of `a`) × (code scale of `b`), so scales for each row of `a` and each column of `b` broadcast.

| Argument | Type | Meaning |
|---|---|---|
| `q_a` | `torch.Tensor` | The codes of `a` |
| `a_params` | `QuantParams` | Their parameters |
| `q_b` | `torch.Tensor` | The codes of `b` |
| `b_params` | `QuantParams` | Their parameters |

```python
acc, acc_params = matmul_integer(torch.tensor([[1, 2], [3, 4]]), a_params, torch.tensor([[5], [6]]), b_params)
print(acc.flatten(), acc_params.scale)  # tensor([17, 39]) tensor(0.0010)
```

---

## matmul_fake

```python
matmul_fake(a: torch.Tensor, a_params: QuantParams, b: torch.Tensor, b_params: QuantParams) -> tuple[torch.Tensor, QuantParams]
```

The training mode. The exact integer accumulator times its scale, with the gradients of the float product. The inputs must be fake-quantized values. FP inputs become integers on the way (FP4 values are doubled).

| Argument | Type | Meaning |
|---|---|---|
| `a` | `torch.Tensor` | Fake-quantized values of `a` |
| `a_params` | `QuantParams` | Their parameters |
| `b` | `torch.Tensor` | Fake-quantized values of `b` |
| `b_params` | `QuantParams` | Their parameters |

```python
y, y_params = matmul_fake(torch.tensor([[0.1, 0.2]]), a_params, torch.tensor([[0.05], [0.06]]), b_params)
print(y)  # tensor([[0.0170]])
```

---

## matmul_float

```python
matmul_float(a: torch.Tensor, a_params: QuantParams, b: torch.Tensor, b_params: QuantParams) -> tuple[torch.Tensor, QuantParams]
```

The float mode (calibration). The float product, and the parameters that the accumulator has in the other modes.

| Argument | Type | Meaning |
|---|---|---|
| `a`, `b` | `torch.Tensor` | Float values |
| `a_params`, `b_params` | `QuantParams` | Their parameters |

```python
y, y_params = matmul_float(torch.tensor([[0.1, 0.2]]), a_params, torch.tensor([[0.05], [0.06]]), b_params)
print(y)  # tensor([[0.0170]])
```
