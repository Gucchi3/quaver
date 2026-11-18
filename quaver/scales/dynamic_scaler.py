# @file dynamic_scaler.py
# @brief 今のテンソルから求める scale（dynamic quantization）
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats import NumberFormat, build_format, device_define
from .compute  import compute_scale
from .group    import group_merge
from .minmax   import minmax_measure
from .types    import Granularity, GranularityKind



class DynamicScaler(nn.Module):
    '''
    #* 毎回、今のテンソルの範囲を測る（dynamic quantization）。
    #? 重みと、トークンごとに scale を 1 つ持つ活性化に向いている。状態を持たないので、推論でも測る。
    '''
    
    learnable: bool = False
    
    def __init__(self, percentile: float | None = None, symmetric: bool = True) -> None:
        '''
        #* 範囲の設定を持つ。
        #? percentile=None のときは最大を使う。
        '''
        
        super().__init__()
        self.percentile = percentile
        self.symmetric  = symmetric
        
        return None
    
    def forward(self, x: torch.Tensor, number_format: NumberFormat, granularity: Granularity, update: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        #* x の scale と zero point を返す。どちらも x に broadcast できる。
        #? op torch.ops.quaver.dynamic_scale が測るので、書き出したモデルは設定付きのノードを 1 つ残す（デバイスも同じ計算をする）。
        #! これまでどおり、scale には勾配が来ない（x は detach する）。
        '''
        
        return torch.ops.quaver.dynamic_scale(x.detach(), number_format.name, self.percentile, self.symmetric, granularity.kind, granularity.axis, granularity.block_size)


@device_define("dynamic_scale")
def dynamic_scaler_device(x: torch.Tensor, number_format: str, percentile: float | None, symmetric: bool, kind: GranularityKind, axis: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* op "dynamic_scale": 粒度のグループごとに、今の x の範囲（最大か percentile）から scale と zero point を求める。
    #? DynamicScaler.forward() が torch.ops.quaver.dynamic_scale として呼ぶ。どちらの出力も x に broadcast できる。
    #! 計算は float（範囲と割り算）。デバイスにも同じ手順が要るので、この op は書き出したモデルで 1 つのノードのまま残る。
    '''
    
    granularity       = Granularity(kind, axis, block_size)
    low, high         = minmax_measure(x, granularity, percentile, symmetric)
    scale, zero_point = compute_scale(low, high, build_format(number_format), symmetric)
    return group_merge(scale, x.shape, granularity), group_merge(zero_point, x.shape, granularity)
