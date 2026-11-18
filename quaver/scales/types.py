# @file types.py
# @brief scale の型
# @author Kohei Yamaguchi, Claude

from dataclasses import dataclass
from typing      import Literal

import torch

from ..formats import NumberFormat

GranularityKind = Literal["tensor", "channel", "block"]
LsqInit         = Literal["lsq", "gauss", "mse", "range"]
MathDtype       = Literal["float32", "float64"]



@dataclass(frozen=True)
class Granularity:
    '''
    #* テンソルが持つ scale の数を表す: 全部で 1 つ、チャネルごとに 1 つ、ブロックごとに 1 つ。
    #? kind は "tensor"、"channel"（axis の番号ごとに 1 つ）、"block"（axis に沿った block_size 個の値）。
    #! 軸全体を 1 つのブロックにすると、トークンごとに scale を 1 つ持つ。
    '''
    
    kind      : GranularityKind = "tensor"
    axis      : int             = 0
    block_size: int             = 16


@dataclass(frozen=True)
class BlockScale:
    '''
    #* ブロックごとの scale のテンソル（NVFP4 など）のブロックの scale を、正確な整数の計算のために整数で持つ。
    #? codes: 各値のブロックの scale の、scale_format での code（実数のブロックの scale = code x scale_format の step_min）。tensor_scale: テンソルの 2 段目の scale。
    #! 要素の実数の値 = 要素の code x その形式の step_min x ブロックの scale x tensor_scale。
    '''
    
    codes       : torch.Tensor
    scale_format: NumberFormat
    tensor_scale: torch.Tensor


@dataclass(frozen=True)
class QuantParams:
    '''
    #* 量子化したテンソルの scale、zero point、数値形式を持つ。
    #? 実数の値 = (code - zero_point) x code_scale、code_scale = scale x 形式の step_min。
    #! ブロックごとの scale のテンソルでは block を決める: そのとき scale は各値のブロックの scale x テンソルの scale で、block はその整数の部分を持つ。
    #! NEQ の出力では peak を決める: デバイスが持つ整数 p（scale = p / (2^NORM_BITS x 形式の一番大きな値)）。
    '''
    
    scale        : torch.Tensor
    zero_point   : torch.Tensor
    number_format: NumberFormat
    block        : BlockScale | None   = None
    peak         : torch.Tensor | None = None
    
    @property
    def code_scale(self) -> torch.Tensor:
        '''
        #* 整数の code の 1 刻みの実数の値を返す。
        '''
        
        return self.scale * self.number_format.step_min
