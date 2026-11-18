# @file pool.py
# @brief 整数の code での平均の pooling
# @author Kohei Yamaguchi, Claude

from torch.nn import functional as F

import math
import torch

from ..formats import RoundMode, device_define, ste_attach
from ..scales  import QuantParams, params_codes, params_math_dtype, params_named, params_note, params_round, params_values



def pool_avg2d_fake(x: torch.Tensor, x_params: QuantParams, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None, mode: RoundMode) -> torch.Tensor:
    '''
    #* training mode の平均の pooling: pool_avg2d_integer と同じ整数の計算で、勾配はそのまま通す。
    #? 計算には params_math_dtype() の型を使う。
    '''
    
    dtype = params_math_dtype()
    codes = _pool_avg2d_codes(params_codes(x, x_params, dtype), x_params, kernel_size, stride, mode, dtype)
    return ste_attach(F.avg_pool2d(x, kernel_size, stride), params_values(codes, x_params, dtype))


def pool_avg2d_integer(q_x: torch.Tensor, x_params: QuantParams, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None, mode: RoundMode) -> torch.Tensor:
    '''
    #* integer mode の平均の pooling: 各窓の code を足し、個数で割って丸める。
    #? 出力は、入力の scale と zero point をそのまま持つ。デバイスの op torch.ops.quaver.pool_avg2d が計算する。
    '''
    
    window = list(kernel_size) if isinstance(kernel_size, tuple) else [kernel_size, kernel_size]
    step   = window if stride is None else list(stride) if isinstance(stride, tuple) else [stride, stride]
    return params_note(torch.ops.quaver.pool_avg2d(q_x, x_params.zero_point.to(torch.int64), x_params.number_format.name, window, step, mode), x_params)


@device_define("pool_avg2d")
def pool_avg2d_device(q_x: torch.Tensor, zero_point: torch.Tensor, number_format: str, kernel_size: list[int], stride: list[int], mode: RoundMode) -> torch.Tensor:
    '''
    #* デバイスの op "pool_avg2d": 各窓の code（zero point を引いたもの）を足し、個数で割って形式に丸め、zero point を足し戻す。
    #? pool_avg2d_integer() が torch.ops.quaver.pool_avg2d として呼ぶ。計算は float64: 和は正確な整数で、割り算は正しく丸める。
    '''
    
    return _pool_avg2d_codes(q_x, params_named(zero_point, number_format), tuple(kernel_size), tuple(stride), mode, torch.float64)


def pool_avg2d_float(x: torch.Tensor, x_params: QuantParams, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None, mode: RoundMode) -> torch.Tensor:
    '''
    #* float のモード（calibration）: 丸めずに、float の平均の pooling をする。
    '''
    
    return F.avg_pool2d(x, kernel_size, stride)


def _pool_avg2d_codes(q_x: torch.Tensor, x_params: QuantParams, kernel_size: int | tuple[int, ...], stride: int | tuple[int, ...] | None, mode: RoundMode, dtype: torch.dtype) -> torch.Tensor:
    '''
    #* 各窓の中心をそろえた code を足し、個数で割って、形式に丸め直す。
    #? pool_avg2d_fake() と pool_avg2d_device() が使うので、2 つのモードは一致する。x_params は、zero point と形式だけを使う。
    #! この整数の割り算は正しく丸めるので、どの丸め方でも正確である（和が 2^24 より小さい間は float32 でも）。
    '''
    
    # 各窓の、中心をそろえた code を足す
    centered = (q_x - x_params.zero_point.to(torch.int64)).to(dtype)
    total    = F.avg_pool2d(centered, kernel_size, stride, divisor_override=1)
    count    = math.prod(kernel_size) if isinstance(kernel_size, tuple) else kernel_size * kernel_size
    
    # 平均を求め、形式に丸め直す
    codes, _ = params_round(total / count, x_params, mode)
    return codes
