# @file tiny_net.py
# @brief The small CIFAR-10 model of the guides, for the flow page of the API reference
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from quant_depthwise_conv2d import QuantDepthwiseConv2d
from quaver                 import Granularity, QuantAvgPool2d, QuantConv2d, Quantizer, QuantLinear, QuantParams, RangeScaler, build_format



class TinyNet(nn.Module):
    '''
    #* The small CIFAR-10 model of the guides.
    #? flow_trace.py runs it in each mode and records its calls.
    '''
    
    def __init__(self) -> None:
        '''
        #* Make the layers.
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
        #* Run the model: fake-quantized values in the training mode, and codes in the integer mode.
        '''
        
        x, params = self.quantizer.encode(images) if self.integer_mode else self.quantizer(images)
        x, params = self.conv1(x, params)
        x, params = self.pointwise2(*self.depthwise2(x, params))
        x, params = self.pool(x, params)
        x, params = self.classifier(x.flatten(1), params)
        return (x, params) if self.integer_mode else x
