# @file nvfp4_basic_block.py
# @brief Block of ResNet-8 with block-scaled convs (NVFP4 and the like), with integer math only
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from quaver     import Granularity, QuantAdd, QuantBlockConv2d, Quantizer, QuantParams, RangeScaler, build_format
from quaver.ops import mode_ops
from .types     import Resnet8Setting



class Nvfp4BasicBlock(nn.Module):
    '''
    #* BasicBlock with block-scaled convs: conv1 with ReLU into block-scaled values, conv2 and the shortcut into INT8, an INT8 residual add, and then ReLU and UINT8.
    #? The block input is UINT8 (the residual), and conv1 and the shortcut quantize it to blocks themselves. fp4 weights, ufp4 activations, and ue4m3 block scales are NVFP4.
    #! conv1, conv2, and shortcut have the names of BasicBlock, so its folded weights load as they are.
    '''
    
    def __init__(self, in_channels: int, out_channels: int, stride: int, setting: Resnet8Setting) -> None:
        '''
        #* Make the two 3x3 convs, the 1x1 shortcut when the shape changes, the add, and the quantizer of the block output.
        '''
        
        # conv1 gives block-scaled values (after ReLU) to conv2, and conv2 and the shortcut give INT8 to the add
        super().__init__()
        self.conv1    = QuantBlockConv2d(in_channels, out_channels, 3, stride, 1, weight_format=setting.weight_format, scale_format=setting.scale_format, input_format=setting.act_format, output_format=setting.act_format, output_block=True,
                                         activation="relu")
        self.conv2    = QuantBlockConv2d(out_channels, out_channels, 3, 1, 1, weight_format=setting.weight_format, scale_format=setting.scale_format, output_format="int8")
        self.shortcut = None if stride == 1 and in_channels == out_channels else QuantBlockConv2d(in_channels, out_channels, 1, stride, 0, weight_format=setting.weight_format, scale_format=setting.scale_format, input_format=setting.act_format,
                                                                                                  output_format="int8")
        
        # INT8 residual add, and the quantizer of the UINT8 block output (after ReLU)
        self.add              = QuantAdd("int8")
        self.output_quantizer = Quantizer(build_format("uint8"), RangeScaler(), Granularity(), batch_axis=0)
        self.integer_mode     = False
        self.float_mode       = False
        
        return None
    
    def forward(self, x: torch.Tensor, x_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* Run the block in the current mode (fake values, integer codes, or float values): UINT8(ReLU(INT8(conv2(blocks(ReLU(conv1(x))))) + shortcut(x))).
        '''
        
        # Shortcut: the input as it is, or the 1x1 conv into INT8
        identity, identity_params = (x, x_params) if self.shortcut is None else self.shortcut(x, x_params)
        
        # conv1 ends in block-scaled values (after ReLU), and conv2 in INT8
        y, y_params = self.conv1(x, x_params)
        y, y_params = self.conv2(y, y_params)
        
        # INT8 residual add, then ReLU and UINT8 (the integer math on the codes)
        ops         = mode_ops(self.integer_mode, self.float_mode)
        y, y_params = self.add(y, y_params, identity, identity_params)
        y           = ops.activation(y, y_params, "relu")
        out_params  = self.output_quantizer.params(y) if self.integer_mode else self.output_quantizer.measure(y)
        return ops.requantize(y, y_params, out_params, "mul16_shift"), out_params
