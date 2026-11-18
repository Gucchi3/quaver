# @file ste.py
# @brief 丸めと量子化の straight-through の勾配（STE）
# @author Kohei Yamaguchi, Claude

import torch

from .format import format_quantize
from .round  import build_round
from .types  import NumberFormat, RoundMode



def ste_round(x: torch.Tensor, mode: RoundMode) -> torch.Tensor:
    '''
    #* forward では x を丸め、backward では勾配をそのまま通す。
    '''
    
    return build_round(mode)(x).detach() + (x - x.detach())


def ste_quantize(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, number_format: NumberFormat, mode: RoundMode) -> torch.Tensor:
    '''
    #* scale と zero point で x を fake quantize する。勾配は範囲の中だけを通る。
    #? Quantizer が、学習しない scaler のために使う。
    #! scale と zero point には勾配が来ない。
    '''
    
    # 形式のグリッドに丸め、zero point を足す
    scale      = scale.detach()
    zero_point = zero_point.detach()
    scaled     = x / scale
    grid       = format_quantize(scaled, number_format, mode, zero_point)
    inside     = (scaled + zero_point >= number_format.min_value) & (scaled + zero_point <= number_format.max_value)
    
    # 実数の値に戻し、勾配をそのまま通す
    return ste_attach(x, (grid - zero_point) * scale, inside)


def ste_attach(reference: torch.Tensor, value: torch.Tensor, inside: torch.Tensor | None = None) -> torch.Tensor:
    '''
    #* forward では value を返し、backward では reference の勾配を使う。
    #? fake の op が、整数の計算の正確な結果に勾配を与えるのに使う。
    #! inside が False の所では、勾配は 0 になる。forward の値は value と完全に同じ（float の誤差はない）。
    '''
    
    attached = value.detach() + (reference - reference.detach())
    if inside is None:
        return attached
    return torch.where(inside, attached, value.detach())
