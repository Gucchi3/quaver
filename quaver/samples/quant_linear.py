# @file quant_linear.py
# @brief 量子化した Linear のサンプル
# @author Kohei Yamaguchi, Claude

from torch    import nn
from torch.nn import functional as F

import torch

from ..formats    import RoundMode
from ..ops        import ActivationKind, accumulate_linear
from ..quantizers import Scaler
from ..scales     import QuantParams
from .quant_layer import QuantLayer



class QuantLinear(QuantLayer):
    '''
    #* 量子化した重み（チャネルごとに scale を 1 つ）、INT32 のバイアス、必要なら ReLU か ReLU6、必要なら requantize を持つ Linear。
    #? forward(x, x_params) は (出力, 出力のパラメータ) を返す。output_format がないときは、出力は INT32 の累積になる。
    #! forward() が受け取るものと返すものが違うので、nn.Linear の子クラスにはしない。重みとバイアスは同じ名前にする。
    '''
    
    output_axis = -1
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, weight_format: str = "int8", output_format: str | None = None, activation: ActivationKind = "none", style: str = "mul16_shift", round_mode: RoundMode | None = None,
                 weight_symmetric: bool = True, output_symmetric: bool = True, output_scaler: Scaler | None = None) -> None:
        '''
        #* linear の層とその量子化器を作る。
        #? activation は "none"、"relu"、"relu6" のどれか。style はデバイスの requantize の方法: mul16_shift、shift_only、exact。output_scaler は出力の scale を共有する（scale の共有のための SharedScaler）。
        #! weight_symmetric=False か output_symmetric=False のときは、その量子化器に zero point を持たせる（INT の形式だけ）。
        '''
        
        super().__init__(nn.Linear(in_features, out_features, bias=bias), weight_format, output_format, activation, style, round_mode, weight_symmetric, output_symmetric, output_scaler)
        
        return None
    
    def _layer_float(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        '''
        #* training mode と float のモード: fake quantize した（または float の）値で float の linear の op をする。
        '''
        
        return F.linear(x, weight, bias)
    
    def _layer_integer(self, q_x: torch.Tensor, x_params: QuantParams, q_w: torch.Tensor, w_params: QuantParams) -> torch.Tensor:
        '''
        #* integer mode: code での正確な linear の累積。
        '''
        
        return accumulate_linear(q_x, x_params, q_w, w_params)
