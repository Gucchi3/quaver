# @file types.py
# @brief 量子化した op の型
# @author Kohei Yamaguchi, Claude

from collections.abc import Callable
from dataclasses     import dataclass
from typing          import Literal

import torch

from ..formats import RoundMode

ActivationKind = Literal["none", "relu", "relu6"]
NeqScope       = Literal["token", "tensor"]



@dataclass(frozen=True)
class RequantizeStyle:
    '''
    #* デバイスが累積の scale を直す方法を表す: multiplier の上限、shift の上限、丸め方。
    #? multiplier_max = 0 は "exact" の意味: multiplier と shift でなく、実数の比を float64 で持つ。
    #! 積（累積 x multiplier）は全部の幅で持つ。16 bit の multiplier なら 32 + 16 bit。
    '''
    
    name          : str
    multiplier_max: int
    shift_max     : int
    round_mode    : RoundMode


@dataclass(frozen=True)
class FixedPoint:
    '''
    #* scale の比を表す multiplier と shift を持つ: ratio = multiplier / 2^shift。
    #? "exact" の style は、比そのものを multiplier に入れ（float64）、shift を 0 にする。
    '''
    
    multiplier: torch.Tensor
    shift     : torch.Tensor
    style     : RequantizeStyle


@dataclass(frozen=True)
class OpSet:
    '''
    #* すべての op の 1 つの版（training、integer、float）を持ち、層が forward を 1 回だけ書けるようにする。
    #? mode_ops() が、モードの組を返す。1 つの op の各版は同じ引数を受け取る。
    '''
    
    requantize         : Callable
    add                : Callable
    pool_avg2d         : Callable
    matmul             : Callable
    normalize_l1       : Callable
    normalize_l1_scores: Callable
    activation         : Callable
    block_quantize     : Callable
