# @file nvfp4_resnet8.py
# @brief ブロックの中でブロックごとの scale の conv（NVFP4 など）を使う ResNet-8 で、整数の計算だけで動く
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from quaver             import QuantAvgPool2d, QuantConv2d, QuantLinear, QuantParams, build_format
from .nvfp4_basic_block import Nvfp4BasicBlock
from .types             import Resnet8Setting



class Nvfp4Resnet8(nn.Module):
    '''
    #* ブロックの中でブロックごとの scale の conv を使う ResNet-8（NVFP4: fp4 の重みと ufp4 の入力で、16 個のブロックごとに ue4m3 の scale を持つ）。ブロックの外は、静的な scale の INT8 か UINT8 の code にする。
    #? stem と分類器の重みは INT8（入力の層と出力の層）。残差は UINT8 で、ReLU、足し算、pool は code の上で整数の計算をする。
    #! integer mode では、QuantResnet8 と同じく、どの層も整数で計算する。層の名前は Resnet8 と同じなので、畳み込んだ FP32 の重みをそのまま読み込める。
    '''
    
    input_scale: torch.Tensor
    
    def __init__(self, setting: Resnet8Setting) -> None:
        '''
        #* stem、ブロック、pool、分類器を作る。
        #? setting がブロックの形式を決める: 重みと活性化の要素の形式と、ブロックの scale の形式。
        '''
        
        super().__init__()
        self.stem_conv    = QuantConv2d(3, 16, 3, 1, 1, weight_format="int8", output_format="uint8", activation="relu")
        self.stack1       = Nvfp4BasicBlock(16, 16, 1, setting)
        self.stack2       = Nvfp4BasicBlock(16, 32, 2, setting)
        self.stack3       = Nvfp4BasicBlock(32, 64, 2, setting)
        self.pool         = QuantAvgPool2d(8)
        self.classifier   = QuantLinear(64, 10, weight_format="int8")
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
        
        # INT8 の重みの stem、ブロックごとの scale の conv を持つブロック 3 つ、UINT8 の code の pool、INT8 の重みの分類器
        x, params = self.stem_conv(x, params)
        x, params = self.stack1(x, params)
        x, params = self.stack2(x, params)
        x, params = self.stack3(x, params)
        x, params = self.pool(x, params)
        x, params = self.classifier(x.flatten(1), params)
        return (x, params) if self.integer_mode else x
