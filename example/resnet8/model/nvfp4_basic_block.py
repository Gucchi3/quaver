# @file nvfp4_basic_block.py
# @brief ブロックごとの scale の conv（NVFP4 など）を使う ResNet-8 のブロックで、整数の計算だけで動く
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from quaver     import Granularity, QuantAdd, QuantBlockConv2d, Quantizer, QuantParams, RangeScaler, build_format
from quaver.ops import mode_ops
from .types     import Resnet8Setting



class Nvfp4BasicBlock(nn.Module):
    '''
    #* ブロックごとの scale の conv を使う BasicBlock。conv1 は ReLU のあとブロックごとの scale の値に、conv2 と shortcut は INT8 に出し、INT8 の残差を足して、ReLU と UINT8 にする。
    #? ブロックの入力は UINT8（残差）で、conv1 と shortcut が自分でブロックに量子化する。fp4 の重み、ufp4 の活性化、ue4m3 のブロックの scale が NVFP4 になる。
    #! conv1、conv2、shortcut の名前は BasicBlock と同じなので、畳み込んだ重みをそのまま読み込める。
    '''
    
    def __init__(self, in_channels: int, out_channels: int, stride: int, setting: Resnet8Setting) -> None:
        '''
        #* 2 つの 3x3 conv、形が変わるときの 1x1 の shortcut、足し算、ブロックの出力の量子化器を作る。
        '''
        
        # conv1 はブロックごとの scale の値（ReLU のあと）を conv2 に渡し、conv2 と shortcut は INT8 を足し算に渡す
        super().__init__()
        self.conv1    = QuantBlockConv2d(in_channels, out_channels, 3, stride, 1, weight_format=setting.weight_format, scale_format=setting.scale_format, input_format=setting.act_format, output_format=setting.act_format, output_block=True,
                                         activation="relu")
        self.conv2    = QuantBlockConv2d(out_channels, out_channels, 3, 1, 1, weight_format=setting.weight_format, scale_format=setting.scale_format, output_format="int8")
        self.shortcut = None if stride == 1 and in_channels == out_channels else QuantBlockConv2d(in_channels, out_channels, 1, stride, 0, weight_format=setting.weight_format, scale_format=setting.scale_format, input_format=setting.act_format,
                                                                                                  output_format="int8")
        
        # INT8 の残差の足し算と、UINT8 のブロックの出力（ReLU のあと）の量子化器
        self.add              = QuantAdd("int8")
        self.output_quantizer = Quantizer(build_format("uint8"), RangeScaler(), Granularity(), batch_axis=0)
        self.integer_mode     = False
        self.float_mode       = False
        
        return None
    
    def forward(self, x: torch.Tensor, x_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* 今のモード（fake の値、整数の code、float の値）でブロックを動かす: UINT8(ReLU(INT8(conv2(blocks(ReLU(conv1(x))))) + shortcut(x)))。
        '''
        
        # shortcut: 入力をそのまま使うか、INT8 に出す 1x1 conv
        identity, identity_params = (x, x_params) if self.shortcut is None else self.shortcut(x, x_params)
        
        # conv1 はブロックごとの scale の値（ReLU のあと）で終わり、conv2 は INT8 で終わる
        y, y_params = self.conv1(x, x_params)
        y, y_params = self.conv2(y, y_params)
        
        # INT8 の残差の足し算、次に ReLU と UINT8（code の上の整数の計算）
        ops         = mode_ops(self.integer_mode, self.float_mode)
        y, y_params = self.add(y, y_params, identity, identity_params)
        y           = ops.activation(y, y_params, "relu")
        out_params  = self.output_quantizer.params(y) if self.integer_mode else self.output_quantizer.measure(y)
        return ops.requantize(y, y_params, out_params, "mul16_shift"), out_params
