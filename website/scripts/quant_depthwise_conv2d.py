# @file quant_depthwise_conv2d.py
# @brief ガイドの depthwise conv。API リファレンスの流れのページ用
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch
import torch.nn.functional as F

from quaver.ops     import ActivationKind, accumulate_conv2d
from quaver.samples import QuantLayer
from quaver.scales  import QuantParams



class QuantDepthwiseConv2d(QuantLayer):
    '''
    #* ガイドと同じく、QuantLayer の上に自分で作った depthwise conv。
    #? tiny_net.py が使い、flow_trace.py がその呼び出しを記録する。
    '''
    
    output_axis = 1
    
    def __init__(self, channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_format: str | None = None, activation: ActivationKind = "none") -> None:
        '''
        #* float の depthwise conv から conv を作る。
        '''
        
        conv = nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels)
        super().__init__(conv, "int8", output_format, activation, "mul16_shift", None, True, True, None)
        
        # conv の設定（groups は書き出しに入る）
        self.stride  = stride
        self.padding = padding
        self.groups  = channels
        
        return None
    
    def _layer_float(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        '''
        #* training mode と float のモードのための、自分の float の depthwise conv。
        '''
        
        return F.conv2d(x, weight, bias, self.stride, self.padding, groups=self.groups)
    
    def _layer_integer(self, q_x: torch.Tensor, x_params: QuantParams, q_w: torch.Tensor, w_params: QuantParams) -> torch.Tensor:
        '''
        #* integer mode のための、code での自分の正確な depthwise の和。
        '''
        
        return accumulate_conv2d(q_x, x_params, q_w, w_params, self.stride, self.padding, 1, self.groups)
