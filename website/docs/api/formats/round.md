---
title: "round.py"
description: "Exact rounding functions, one for each rounding rule, and their registry."
sidebar_position: 5
---

The rounding functions. Each one is exact for every float. [Rounding](../../learn/rounding.md) explains the rules.

```python
from quaver.formats import ROUND_REGISTRY, build_round, round_floor, round_half_away, round_half_even, round_half_up
```

The examples on this page use:

```python
x = torch.tensor([-2.5, -1.5, 0.5, 1.5, 2.5, 2.4])
```

---

## round_half_away

```python
round_half_away(x: torch.Tensor) -> torch.Tensor
```

Rounds to the nearest integer. A half goes away from zero. This is the usual rule of INT formats, like TFLite.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values to round |

```python
print(round_half_away(x))  # tensor([-3., -2.,  1.,  2.,  3.,  2.])
```

---

## round_half_even

```python
round_half_even(x: torch.Tensor) -> torch.Tensor
```

Rounds to the nearest integer. A half goes to the even integer. This is the usual rule of FP formats, like IEEE floats and `torch.round()`.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values to round |

```python
print(round_half_even(x))  # tensor([-2., -2.,  0.,  2.,  2.,  2.])
```

---

## round_half_up

```python
round_half_up(x: torch.Tensor) -> torch.Tensor
```

Rounds to the nearest integer. A half goes up, toward +∞. This is what integer hardware does when it adds half a step before a right shift.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values to round |

```python
print(round_half_up(x))  # tensor([-2., -1.,  1.,  2.,  3.,  2.])
```

---

## round_floor

```python
round_floor(x: torch.Tensor) -> torch.Tensor
```

Rounds down, like an arithmetic right shift.

| Argument | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | The values to round |

```python
print(round_floor(x))  # tensor([-3., -2.,  0.,  1.,  2.,  2.])
```

---

## ROUND_REGISTRY

```python
ROUND_REGISTRY: dict[str, Callable[[torch.Tensor], torch.Tensor]]
```

The rounding functions by name: `"half_away"`, `"half_even"`, `"half_up"`, and `"floor"`. To add a rule, write its function, add it here, and add its name to `RoundMode`.

```python
print(list(ROUND_REGISTRY))  # ['half_away', 'half_even', 'half_up', 'floor']
```

---

## build_round

```python
build_round(name: str) -> Callable[[torch.Tensor], torch.Tensor]
```

Returns the rounding function with this name. An unknown name raises `KeyError` with the list of names.

| Argument | Type | Meaning |
|---|---|---|
| `name` | `str` | A name from `ROUND_REGISTRY` |

```python
round_fn = build_round("half_even")
print(round_fn(torch.tensor([2.5, 3.5])))  # tensor([2., 4.])
```
