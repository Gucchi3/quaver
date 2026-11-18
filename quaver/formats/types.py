# @file types.py
# @brief 数値形式の型
# @author Kohei Yamaguchi, Claude

from dataclasses import dataclass
from typing      import Literal

RoundMode = Literal["half_away", "half_even", "half_up", "floor"]



@dataclass(frozen=True)
class NumberFormat:
    '''
    #* 1 つの数値形式（INT か FP）の詳しい情報を、値を scale で割ったあとのグリッドで持つ。
    #? integer_format() か fp_format() で作るか、build_format() で名前から受け取る。
    '''
    
    name         : str
    kind         : str
    bits         : int
    signed       : bool
    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int
    nan_top      : bool
    min_value    : float
    max_value    : float
    step_min     : float
