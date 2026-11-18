# @file matmul.py
# @brief 量子化した 2 つのテンソルの行列積（attention 用）
# @author Kohei Yamaguchi, Claude

import torch

from ..formats   import ste_attach
from ..scales    import QuantParams, params_codes, params_math_dtype, params_values
from .accumulate import ACCUMULATOR_FORMAT, accumulate_matmul, accumulate_product



def matmul_fake(a: torch.Tensor, a_params: QuantParams, b: torch.Tensor, b_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
    '''
    #* training mode の行列積: 正確な整数の累積 x その scale で、勾配は float で求める。
    #? 入力は fake quantize した値でなければならない。FP の入力は途中で整数になる（FP4 の値は 2 倍になる）。
    #! code を取り出し、params_math_dtype() の型で値に戻す。累積そのものはいつも正確（float64）。
    '''
    
    dtype       = params_math_dtype()
    acc, params = matmul_integer(params_codes(a, a_params, dtype), a_params, params_codes(b, b_params, dtype), b_params)
    return ste_attach(accumulate_product(a, b), params_values(acc, params, dtype)), params


def matmul_integer(q_a: torch.Tensor, a_params: QuantParams, q_b: torch.Tensor, b_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
    '''
    #* integer mode の行列積: 正確な累積（int64）とそのパラメータ。
    #? 累積の scale は (a の code の scale) x (b の code の scale) なので、行ごとと列ごとの scale は broadcast される。
    '''
    
    acc   = accumulate_matmul(q_a, a_params, q_b, b_params)
    scale = a_params.code_scale * b_params.code_scale
    return acc, QuantParams(scale, torch.zeros([1] * scale.ndim, device=scale.device), ACCUMULATOR_FORMAT)


def matmul_float(a: torch.Tensor, a_params: QuantParams, b: torch.Tensor, b_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
    '''
    #* float のモード（calibration）: float の行列積と、ほかのモードで累積が持つパラメータ。
    '''
    
    scale = a_params.code_scale * b_params.code_scale
    return accumulate_product(a, b), QuantParams(scale, torch.zeros_like(scale), ACCUMULATOR_FORMAT)
