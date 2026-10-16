# @file quant_depthwise_conv2d.py
# @brief The depthwise conv of the guides, for the flow page of the API reference
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch
import torch.nn.functional as F

from quaver.ops     import ActivationKind, accumulate_conv2d
from quaver.samples import QuantLayer
from quaver.scales  import QuantParams



class QuantDepthwiseConv2d(QuantLayer):
    '''
    #* A depthwise conv of your own, built on QuantLayer, as in the guides.
    #? tiny_net.py uses it, and flow_trace.py records its calls.
    '''
    
    output_axis = 1
    
    def __init__(self, channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_format: str | None = None, activation: ActivationKind = "none") -> None:
        '''
        #* Make the conv from a float depthwise conv.
        '''
        
        conv = nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels)
        super().__init__(conv, "int8", output_format, activation, "mul16_shift", None, True, True, None)
        
        # The settings of the conv (groups goes to the export)
        self.stride  = stride
        self.padding = padding
        self.groups  = channels
        
        return None
    
    def _layer_float(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        '''
        #* Your float depthwise conv, for the training and float modes.
        '''
        
        return F.conv2d(x, weight, bias, self.stride, self.padding, groups=self.groups)
    
    def _layer_integer(self, q_x: torch.Tensor, x_params: QuantParams, q_w: torch.Tensor, w_params: QuantParams) -> torch.Tensor:
        '''
        #* Your exact depthwise sums on codes, for the integer mode.
        '''
        
        return accumulate_conv2d(q_x, x_params, q_w, w_params, self.stride, self.padding, 1, self.groups)
