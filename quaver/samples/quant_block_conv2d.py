# @file quant_block_conv2d.py
# @brief 整数の計算だけで動く、ブロックごとの scale の Conv2d（NVFP4 など）のサンプル
# @author Kohei Yamaguchi, Claude

from torch    import nn
from torch.nn import functional as F

import torch

from ..ops              import ActivationKind, accumulate_conv2d
from ..quantizers       import Scaler
from ..scales           import QuantParams
from .quant_block_layer import QuantBlockLayer



class QuantBlockConv2d(QuantBlockLayer):
    '''
    #* ブロックごとの scale の重み（入力の 16 チャネルのブロックごとに scale を 1 つ）、必要ならブロックごとの scale の入力、正確な wide の累積、次の op の形式の出力を持つ Conv2d。
    #? scale_format が "ue4m3" の fp4 の重みが NVFP4 になる。input_format "ufp4" は入力をブロックに量子化する（ReLU のあとの入力用）。forward(x, x_params) は (出力, 出力のパラメータ) を返す。
    #! forward() が受け取るものと返すものが違うので、nn.Conv2d の子クラスにはしない。重みとバイアスは同じ名前にする。
    '''
    
    output_axis = 1
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True, weight_format: str = "fp4", scale_format: str = "ue4m3", input_format: str | None = None, output_format: str = "int8",
                 output_block: bool = False, activation: ActivationKind = "none", style: str = "mul16_shift", block_size: int = 16, output_scaler: Scaler | None = None) -> None:
        '''
        #* conv とその量子化器を作る。
        #? input_format=None のときは、入力をそのまま使う（ブロックごとの scale か、INT8）。output_block=True のときは、次のブロックごとの scale の層のために、output_format のブロックごとの scale の出力を出す。
        #? output_scaler は、ブロックの残差のストリームと同じく（scale の共有）、INT の出力（output_block=False）の scale を共有する。
        #! in_channels は block_size の倍数でなければならない。output_block=True なら out_channels も同じ。
        '''
        
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias), weight_format, scale_format, input_format, output_format, output_block, activation, style, block_size, output_scaler)
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
        #* code での正確な conv の累積（各 code に、そのブロックの scale の code をかける）。
        '''
        
        return accumulate_conv2d(q_x, x_params, q_w, w_params, self.stride, self.padding)
