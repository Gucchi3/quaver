# @file quant_resnet8.py
# @brief quaver のサンプルで作った、量子化した ResNet-8
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from quaver             import QuantAvgPool2d, QuantConv2d, QuantLinear, QuantParams, build_format
from .quant_basic_block import QuantBasicBlock
from .types             import Resnet8Setting



class QuantResnet8(nn.Module):
    '''
    #* 量子化した ResNet-8。入力は UINT8（scale 1/255）、stem と分類器の重みは INT8、ブロックの重みと ReLU の出力は setting の形式、残差は INT8 で、code の上で平均の pool をする。
    #? 層の名前は Resnet8 と同じなので、畳み込んだ FP32 の重みをそのまま読み込める。switch_integer_mode() でモデル全体を integer mode にする。
    #! どちらのモードでも実数の logits を返す。integer mode では、分類器の正確な INT32 の累積から求める。
    '''
    
    input_scale: torch.Tensor
    
    def __init__(self, setting: Resnet8Setting) -> None:
        '''
        #* stem、ブロック、pool、分類器を作る。
        #? stem と分類器の重みはいつも INT8 で、その zero point はほかの層と同じく setting に従う。
        '''
        
        super().__init__()
        self.stem_conv    = QuantConv2d(3, 16, 3, 1, 1, weight_format="int8", output_format=setting.act_format, activation="relu", weight_symmetric=setting.weight_symmetric, output_symmetric=setting.act_symmetric)
        self.stack1       = QuantBasicBlock(16, 16, 1, setting)
        self.stack2       = QuantBasicBlock(16, 32, 2, setting)
        self.stack3       = QuantBasicBlock(32, 64, 2, setting)
        self.pool         = QuantAvgPool2d(8)
        self.classifier   = QuantLinear(64, 10, weight_format="int8", weight_symmetric=setting.weight_symmetric)
        self.integer_mode = False
        self.register_buffer("input_scale", torch.tensor(1 / 255))
        
        return None
    
    def forward(self, images: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, QuantParams]:
        '''
        #* training mode では uint8 の画像の logits を返し、integer mode では logits の INT32 の code とそのパラメータを返す。
        #? integer mode はデバイスと同じく code を出すので、書き出したモデルは分類器で終わる。params_values() でそれを logits に変える。
        '''
        
        # 入力は scale 1/255 の uint8 の code: training mode では fake の値、integer mode では code
        params = QuantParams(self.input_scale, torch.zeros_like(self.input_scale), build_format("uint8"))
        x      = images.long() if self.integer_mode else images.float() * self.input_scale
        
        # stem、ブロック 3 つ、pool、分類器
        x, params = self.stem_conv(x, params)
        x, params = self.stack1(x, params)
        x, params = self.stack2(x, params)
        x, params = self.stack3(x, params)
        x, params = self.pool(x, params)
        x, params = self.classifier(x.flatten(1), params)
        return (x, params) if self.integer_mode else x
