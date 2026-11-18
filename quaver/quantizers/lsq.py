# @file lsq.py
# @brief LSQ の fake quantization と、その勾配の scale
# @author Kohei Yamaguchi, Claude

import math
import torch

from ..formats import NumberFormat, RoundMode, format_quantize



def lsq_quantize(x: torch.Tensor, step: torch.Tensor, zero_point: torch.Tensor, number_format: NumberFormat, mode: RoundMode, grad_scale: float) -> torch.Tensor:
    '''
    #* 学習する step（LSQ）と学習する zero point（LSQ+）で x を fake quantize する。
    #? autograd が LSQ の勾配を与える: 範囲の中では d/dstep = round(v) - v、範囲の外では範囲の端。
    #! encode() と同じく、code は zero point を足す前に丸める。float32 で先に zero point を足すと、ちょうど半分の隣の値がちょうど半分に動くことがある。
    '''
    
    # step で割り、LSQ の勾配の scale を step にかける
    step   = lsq_scale_gradient(step, grad_scale)
    scaled = x / step
    
    # 値は丸めて（次に zero point を足し、切り詰めて）求め、勾配は切り詰めた値から straight-through の道で求める
    clipped = (scaled + zero_point).clamp(number_format.min_value, number_format.max_value)
    grid    = format_quantize(scaled, number_format, mode, zero_point).detach() + (clipped - clipped.detach())
    return (grid - zero_point) * step


def lsq_grad_scale(count: int, number_format: NumberFormat) -> float:
    '''
    #* LSQ の勾配の scale 1 / sqrt(N x Qp) を返す。
    #? N は 1 つの step を共有する値の数。Qp は形式の一番大きな値。
    '''
    
    return 1.0 / math.sqrt(max(count, 1) * number_format.max_value)


def lsq_scale_gradient(x: torch.Tensor, factor: float) -> torch.Tensor:
    '''
    #* x の値はそのまま保ち、勾配だけに factor をかける。
    '''
    
    scaled = x * factor
    return x.detach() + (scaled - scaled.detach())
