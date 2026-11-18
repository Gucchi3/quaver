# @file tiny_net.py
# @brief ガイドの小さな CIFAR-10 のモデル。API リファレンスの流れのページ用
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from quant_depthwise_conv2d import QuantDepthwiseConv2d
from quaver                 import Granularity, QuantAvgPool2d, QuantConv2d, Quantizer, QuantLinear, QuantParams, RangeScaler, build_format



class TinyNet(nn.Module):
    '''
    #* ガイドの小さな CIFAR-10 のモデル。
    #? flow_trace.py が各モードで動かし、その呼び出しを記録する。
    '''
    
    def __init__(self) -> None:
        '''
        #* 層を作る。
        '''
        
        super().__init__()
        self.quantizer    = Quantizer(build_format("int8"), RangeScaler(), Granularity(), batch_axis=0)
        self.conv1        = QuantConv2d(3, 32, 3, padding=1, output_format="uint8", activation="relu")
        self.depthwise2   = QuantDepthwiseConv2d(32, 3, stride=2, padding=1, output_format="uint8", activation="relu")
        self.pointwise2   = QuantConv2d(32, 64, 1, output_format="uint8", activation="relu")
        self.pool         = QuantAvgPool2d(16)
        self.classifier   = QuantLinear(64, 10)
        self.integer_mode = False
        
        return None
    
    def forward(self, images: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, QuantParams]:
        '''
        #* モデルを動かす: training mode では fake quantize した値、integer mode では code を使う。
        '''
        
        x, params = self.quantizer.encode(images) if self.integer_mode else self.quantizer(images)
        x, params = self.conv1(x, params)
        x, params = self.pointwise2(*self.depthwise2(x, params))
        x, params = self.pool(x, params)
        x, params = self.classifier(x.flatten(1), params)
        return (x, params) if self.integer_mode else x
