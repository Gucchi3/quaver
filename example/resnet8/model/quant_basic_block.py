# @file quant_basic_block.py
# @brief quaver のサンプルで作った、量子化した ResNet-8 のブロック
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from quaver     import Granularity, QuantAdd, QuantConv2d, Quantizer, QuantParams, RangeScaler, build_format
from quaver.ops import mode_ops
from .types     import Resnet8Setting



class QuantBasicBlock(nn.Module):
    '''
    #* 量子化した BasicBlock。conv1 は ReLU のあと活性化の形式に、conv2 は INT32 の累積に出し、INT8 の残差を足して、ReLU と活性化の形式にする。
    #? 1x1 の shortcut は INT8 を出す。act_symmetric=False のときは、ブロックのすべての活性化（残差の足し算も）が zero point を持つ。
    #! conv1、conv2、shortcut の名前は BasicBlock と同じなので、畳み込んだ重みをそのまま読み込める。
    '''
    
    def __init__(self, in_channels: int, out_channels: int, stride: int, setting: Resnet8Setting) -> None:
        '''
        #* conv、shortcut、足し算、ブロックの出力の量子化器を作る。
        '''
        
        # conv1: 3x3 conv、ReLU、活性化の形式
        super().__init__()
        self.conv1 = QuantConv2d(in_channels, out_channels, 3, stride, 1, weight_format=setting.weight_format, output_format=setting.act_format, activation="relu", weight_symmetric=setting.weight_symmetric, output_symmetric=setting.act_symmetric)
        
        # conv2: INT32 の累積のままにする 3x3 conv（残差の足し算が scale を合わせ直す）
        self.conv2 = QuantConv2d(out_channels, out_channels, 3, 1, 1, weight_format=setting.weight_format, weight_symmetric=setting.weight_symmetric)
        
        # shortcut: 入力をそのまま使うか、INT8 に出す 1x1 conv
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = QuantConv2d(in_channels, out_channels, 1, stride, weight_format=setting.weight_format, output_format="int8", weight_symmetric=setting.weight_symmetric, output_symmetric=setting.act_symmetric)
        
        # INT8 の残差の足し算と、ブロックの出力（ReLU のあと）の量子化器
        self.add              = QuantAdd("int8", output_symmetric=setting.act_symmetric)
        self.output_quantizer = Quantizer(build_format(setting.act_format), RangeScaler(symmetric=setting.act_symmetric), Granularity(), batch_axis=0)
        self.integer_mode     = False
        self.float_mode       = False
        
        return None
    
    def forward(self, x: torch.Tensor, x_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* 今のモード（fake の値、整数の code、float の値）でブロックを動かす。
        '''
        
        # shortcut: 入力をそのまま使うか、INT8 に出す 1x1 conv
        identity, identity_params = (x, x_params) if self.shortcut is None else self.shortcut(x, x_params)
        
        # conv1 は活性化の形式で終わり、conv2 は INT32 の累積のままにする
        y, y_params = self.conv1(x, x_params)
        y, y_params = self.conv2(y, y_params)
        
        # INT8 の残差の足し算、次に ReLU と活性化の形式（requantize は、先に足し算の zero point を引く）
        ops         = mode_ops(self.integer_mode, self.float_mode)
        y, y_params = self.add(y, y_params, identity, identity_params)
        y           = ops.activation(y, y_params, "relu")
        out_params  = self.output_quantizer.params(y) if self.integer_mode else self.output_quantizer.measure(y)
        return ops.requantize(y, y_params, out_params, "mul16_shift"), out_params
