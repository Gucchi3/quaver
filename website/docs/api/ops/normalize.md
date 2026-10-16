---
title: "normalize.py"
description: "NEQ: L1-normalize each token and quantize it in one step, and requantize the attention map of SimA, with integers only."
sidebar_position: 10
---

The two integer steps of SimA attention: NEQ (Normalization-Equivalent Quantization), which L1-normalizes each token of Q and K and quantizes it in one step, and the requantization of the attention map Q Kᵀ. The device finds every changing multiplier with integer division. [SimA attention](../../learn/sima-attention.md) explains them.

```python
from quaver.ops import NORM_BITS, normalize_l1_device, normalize_l1_fake, normalize_l1_float, normalize_l1_integer, normalize_l1_scores_device, normalize_l1_scores_fake, normalize_l1_scores_float, normalize_l1_scores_integer
```

The examples on this page use the accumulators of Q and K for 1 sample, 2 heads, 4 tokens, and 8 channels:

```python
torch.manual_seed(0)
acc_params = QuantParams(torch.tensor(0.01), torch.tensor(0.0), build_format("int32"))
q_acc      = torch.randint(-500, 500, (1, 2, 4, 8))
k_acc      = torch.randint(-500, 500, (1, 2, 4, 8))
int8       = build_format("int8")
```

---

## NORM_BITS

```python
NORM_BITS: int   # 20
```

The bits of the integer p of NEQ: p = floor(max|v| × $2^{20}$ / L1), from 1 to $2^{20}$. The scale of the NEQ codes is p / ($2^{20}$ × the largest value of the format).

```python
print(NORM_BITS)  # 20
```

---

## normalize_l1_integer

```python
normalize_l1_integer(q_x: torch.Tensor, x_params: QuantParams, number_format: NumberFormat, style_name: str, scope: NeqScope = "tensor") -> tuple[torch.Tensor, QuantParams]
```

The integer mode of NEQ. L1-normalizes each token (the last axis) and quantizes it in one step, with 1/L1 in the scale. With `scope="tensor"` (the paper's deployed setting), each sample and head gets one scale; with `"token"`, each token gets one. The returned `QuantParams` keep the integer p in `.peak`, for the attention map.

| Argument | Type | Meaning |
|---|---|---|
| `q_x` | `torch.Tensor` | The codes (usually the accumulator of the Q or K conv), (…, tokens, channels) |
| `x_params` | `QuantParams` | Their parameters (one scale for each channel is fine) |
| `number_format` | `NumberFormat` | The format of the output |
| `style_name` | `str` | The requantization style |
| `scope` | `NeqScope` | `"tensor"` or `"token"` |

```python
q, q_params = normalize_l1_integer(q_acc, acc_params, int8, "mul16_shift")
print(q[0, 0, 0], q_params.peak.flatten())
# tensor([-111,  -64,  106,   63,  113,  -30,  -18,    1]) tensor([263629, 240561])   one p for each head
```

---

## normalize_l1_fake

```python
normalize_l1_fake(x: torch.Tensor, x_params: QuantParams, number_format: NumberFormat, style_name: str, scope: NeqScope = "tensor") -> tuple[torch.Tensor, QuantParams]
```

The training mode of NEQ. The same integer math as `normalize_l1_integer()`, on fake-quantized values. The gradients follow the float normalization. Turn off TF32, so the accumulator codes come back exactly.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | Fake-quantized values (an accumulator) |
| `x_params` | `QuantParams` | Their parameters |
| `number_format` | `NumberFormat` | The format of the output |
| `style_name` | `str` | The requantization style |
| `scope` | `NeqScope` | `"tensor"` or `"token"` |

```python
values, value_params = normalize_l1_fake(q_acc.float() * 0.01, acc_params, int8, "mul16_shift")
# values / value_params.scale gives the same codes as normalize_l1_integer()
```

---

## normalize_l1_float

```python
normalize_l1_float(x: torch.Tensor, x_params: QuantParams, number_format: NumberFormat, style_name: str, scope: NeqScope = "tensor") -> tuple[torch.Tensor, QuantParams]
```

The float mode (calibration). The float L1 normalization of each token, and the scale that NEQ would give it.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | Float values |
| `x_params` | `QuantParams` | Their parameters |
| `number_format` | `NumberFormat` | The format of the output |
| `style_name` | `str` | Not used |
| `scope` | `NeqScope` | `"tensor"` or `"token"` |

```python
values, value_params = normalize_l1_float(q_acc.float() * 0.01, acc_params, int8, "mul16_shift")
print(values[0, 0, 0, :3])  # tensor([-0.2203, -0.1261,  0.2092])
```

---

## normalize_l1_device

```python
normalize_l1_device(q_x: torch.Tensor, zero_point: torch.Tensor, multiplier: torch.Tensor, shift: torch.Tensor, out_format: str, style_name: str, scope: NeqScope) -> tuple[torch.Tensor, torch.Tensor]
```

The device op `normalize_l1` (`torch.ops.quaver.normalize_l1`): puts the channels on one scale (fixed $M_0$ and $n$), measures each token with integers, and rescales each value with the integer multiplier of NEQ. Returns the codes and the integer p. `normalize_l1_integer()` calls it.

| Argument | Type | Meaning |
|---|---|---|
| `q_x` | `torch.Tensor` | The codes |
| `zero_point` | `torch.Tensor` | Their zero point |
| `multiplier`, `shift` | `torch.Tensor` | The fixed $M_0$ and $n$ that put each channel on the largest channel scale |
| `out_format` | `str` | The name of the output format |
| `style_name` | `str` | The requantization style |
| `scope` | `NeqScope` | `"tensor"` or `"token"` |

```python
print(torch.ops.quaver.normalize_l1)  # quaver.normalize_l1
```

---

## normalize_l1_scores_integer

```python
normalize_l1_scores_integer(acc: torch.Tensor, x_params: QuantParams, q_params: QuantParams, k_params: QuantParams, out_params: QuantParams, factor: float, style_name: str) -> torch.Tensor
```

The integer mode of the attention map. Requantizes the exact product Q Kᵀ of two NEQ outputs to the output codes. The ratio is $p_q \cdot p_k \cdot H$, where $p_q$ and $p_k$ are the integers p of NEQ, and $H$ = factor / (top of Q × top of K × output code scale) is fixed. The device cuts $p_q \cdot p_k \cdot M_{0,H}$ to a 16-bit multiplier and a shift with integers.

| Argument | Type | Meaning |
|---|---|---|
| `acc` | `torch.Tensor` | The exact product Q Kᵀ (from `matmul_integer()`) |
| `x_params` | `QuantParams` | Its parameters |
| `q_params` | `QuantParams` | The parameters of Q (with `.peak`) |
| `k_params` | `QuantParams` | The parameters of Kᵀ (with `.peak`) |
| `out_params` | `QuantParams` | The parameters of the attention map |
| `factor` | `float` | $1 / \sqrt{d}$ in SimA |
| `style_name` | `str` | The requantization style |

```python
k, k_params = normalize_l1_integer(k_acc, acc_params, int8, "mul16_shift")
k_t_params  = QuantParams(k_params.scale.transpose(-1, -2), k_params.zero_point.transpose(-1, -2), int8, peak=k_params.peak.transpose(-1, -2))
scores, score_params = matmul_integer(q, q_params, k.transpose(-1, -2), k_t_params)
out_params = QuantParams(torch.tensor(1 / 127), torch.tensor(0.0), int8)
attention  = normalize_l1_scores_integer(scores, score_params, q_params, k_t_params, out_params, 8 ** -0.5, "mul16_shift")
print(attention.shape)  # torch.Size([1, 2, 4, 4])
```

---

## normalize_l1_scores_fake

```python
normalize_l1_scores_fake(x: torch.Tensor, x_params: QuantParams, q_params: QuantParams, k_params: QuantParams, out_params: QuantParams, factor: float, style_name: str) -> torch.Tensor
```

The training mode of the attention map. The same integer math as `normalize_l1_scores_integer()`, on the fake product, with straight-through gradients from `x × factor`. A learned output scale gets the LSQ gradient.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The fake product Q Kᵀ (from `matmul_fake()`) |
| `x_params` | `QuantParams` | Its parameters |
| `q_params`, `k_params` | `QuantParams` | The parameters of Q and Kᵀ |
| `out_params` | `QuantParams` | The parameters of the attention map |
| `factor` | `float` | $1 / \sqrt{d}$ |
| `style_name` | `str` | The requantization style |

```python
# The same arguments as normalize_l1_scores_integer(), with fake values in place of codes
```

---

## normalize_l1_scores_float

```python
normalize_l1_scores_float(x: torch.Tensor, x_params: QuantParams, q_params: QuantParams, k_params: QuantParams, out_params: QuantParams, factor: float, style_name: str) -> torch.Tensor
```

The float mode (calibration). The float product times `factor`, without requantizing.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The float product |
| `x_params`, `q_params`, `k_params`, `out_params` | `QuantParams` | Not used |
| `factor` | `float` | $1 / \sqrt{d}$ |
| `style_name` | `str` | Not used |

```python
print(normalize_l1_scores_float(torch.tensor([2.0]), score_params, q_params, k_t_params, out_params, 0.5, "mul16_shift"))  # tensor([1.])
```

---

## normalize_l1_scores_device

```python
normalize_l1_scores_device(acc: torch.Tensor, q_peak: torch.Tensor, k_peak: torch.Tensor, multiplier: torch.Tensor, shift: torch.Tensor, out_zero_point: torch.Tensor, out_format: str, style_name: str) -> torch.Tensor
```

The device op `normalize_l1_scores` (`torch.ops.quaver.normalize_l1_scores`): requantizes Q Kᵀ with $p_q \cdot p_k \cdot M_{0,H}$ cut to $M_0$ and $n$ with integers.

| Argument | Type | Meaning |
|---|---|---|
| `acc` | `torch.Tensor` | The exact product Q Kᵀ |
| `q_peak`, `k_peak` | `torch.Tensor` | The integers p of Q and K (from `normalize_l1_device()`) |
| `multiplier`, `shift` | `torch.Tensor` | The fixed $M_{0,H}$ and $n_H$ |
| `out_zero_point` | `torch.Tensor` | The zero point of the output |
| `out_format` | `str` | The name of the output format |
| `style_name` | `str` | The requantization style |

```python
print(torch.ops.quaver.normalize_l1_scores)  # quaver.normalize_l1_scores
```
