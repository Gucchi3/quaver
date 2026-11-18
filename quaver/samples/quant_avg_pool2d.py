# @file quant_avg_pool2d.py
# @brief 量子化した平均の pooling のサンプル
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats import RoundMode, format_round_mode
from ..ops     import mode_ops
from ..scales  import QuantParams



class QuantAvgPool2d(nn.Module):
    '''
    #* 整数の code での平均の pooling: 足し、個数で割って、丸める。
    #? forward(x, x_params) は (出力, x_params) を返す。scale は変わらない。
    '''
    
    round_mode: RoundMode | None
    
    def __init__(self, kernel_size: int, stride: int | None = None, round_mode: RoundMode | None = None) -> None:
        '''
        #* pooling の設定を持つ。
        #? round_mode=None のときは、入力の形式のいつもの丸め（INT は half_away）を使う。FP の入力はいつも half to even で丸める。
        '''
        
        super().__init__()
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.round_mode   = round_mode
        self.integer_mode = False
        self.float_mode   = False
        
        return None
    
    def forward(self, x: torch.Tensor, x_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* 今のモード（fake の値、整数の code、float の値）で x を pooling する。
        '''
        
        ops = mode_ops(self.integer_mode, self.float_mode)
        return ops.pool_avg2d(x, x_params, self.kernel_size, self.stride, format_round_mode(x_params.number_format, self.round_mode)), x_params
