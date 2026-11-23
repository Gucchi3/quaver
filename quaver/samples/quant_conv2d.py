# @file quant_conv2d.py
# @brief 量子化した Conv2d のサンプル
# @author Kohei Yamaguchi, Claude

from torch    import nn
from torch.nn import functional as F

import torch

from ..formats    import RoundMode
from ..ops        import ActivationKind, accumulate_conv2d
from ..quantizers import Scaler
from ..scales     import QuantParams
from .quant_layer import QuantLayer



class QuantConv2d(QuantLayer):
    '''
    #* 量子化した重み（チャネルごとに scale を 1 つ）、INT32 のバイアス、必要なら ReLU か ReLU6、必要なら requantize を持つ Conv2d。
    #? forward(x, x_params) は (出力, 出力のパラメータ) を返す。output_format がないときは、出力は INT32 の累積になる。
    #! forward() が受け取るものと返すものが違うので、nn.Conv2d の子クラスにはしない。重みとバイアスは同じ名前にする。
    '''
    
    output_axis = 1
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True, weight_format: str = "int8", output_format: str | None = None, activation: ActivationKind = "none",
                 style: str = "mul16_shift", round_mode: RoundMode | None = None, weight_symmetric: bool = True, output_symmetric: bool = True, output_scaler: Scaler | None = None) -> None:
        '''
        #* conv とその量子化器を作る。
        #? activation は "none"、"relu"、"relu6" のどれか。style はデバイスの requantize の方法: mul16_shift、shift_only、exact。output_scaler は出力の scale を共有する（scale の共有のための SharedScaler）。
        #! weight_symmetric=False か output_symmetric=False のときは、その量子化器に zero point を持たせる（INT の形式だけ）。
        '''
        
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias), weight_format, output_format, activation, style, round_mode, weight_symmetric, output_symmetric, output_scaler)
        self.stride  = stride
        self.padding = padding
        
        return None
    
    def _layer_float(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        '''
        #* training mode と float のモード: fake quantize した（または float の）値で float の conv をする。
        '''
        
        return F.conv2d(x, weight, bias, self.stride, self.padding)
    
    def _layer_integer(self, q_x: torch.Tensor, x_params: QuantParams, q_w: torch.Tensor, w_params: QuantParams) -> torch.Tensor:
        '''
        #* integer mode: code での正確な conv の累積。
        '''
        
        return accumulate_conv2d(q_x, x_params, q_w, w_params, self.stride, self.padding)
