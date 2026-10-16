# @file quant_basic_block.py
# @brief Quantized block of ResNet-8, made from the quaver samples
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from quaver     import Granularity, QuantAdd, QuantConv2d, Quantizer, QuantParams, RangeScaler, build_format
from quaver.ops import mode_ops
from .types     import Resnet8Setting



class QuantBasicBlock(nn.Module):
    '''
    #* Quantized BasicBlock: conv1 with ReLU into the activation format, conv2 into its INT32 accumulator, an INT8 residual add, and then ReLU and the activation format.
    #? The 1x1 shortcut gives INT8. With act_symmetric=False, every activation of the block (the residual add too) has a zero point.
    #! conv1, conv2, and shortcut have the names of BasicBlock, so its folded weights load as they are.
    '''
    
    def __init__(self, in_channels: int, out_channels: int, stride: int, setting: Resnet8Setting) -> None:
        '''
        #* Make the convs, the shortcut, the add, and the quantizer of the block output.
        '''
        
        # conv1: 3x3 conv, ReLU, and the activation format
        super().__init__()
        self.conv1 = QuantConv2d(in_channels, out_channels, 3, stride, 1, weight_format=setting.weight_format, output_format=setting.act_format, activation="relu", weight_symmetric=setting.weight_symmetric, output_symmetric=setting.act_symmetric)
        
        # conv2: 3x3 conv that stays an INT32 accumulator (the residual add rescales it)
        self.conv2 = QuantConv2d(out_channels, out_channels, 3, 1, 1, weight_format=setting.weight_format, weight_symmetric=setting.weight_symmetric)
        
        # Shortcut: the input as it is, or a 1x1 conv into INT8
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = QuantConv2d(in_channels, out_channels, 1, stride, weight_format=setting.weight_format, output_format="int8", weight_symmetric=setting.weight_symmetric, output_symmetric=setting.act_symmetric)
        
        # INT8 residual add, and the quantizer of the block output (after ReLU)
        self.add              = QuantAdd("int8", output_symmetric=setting.act_symmetric)
        self.output_quantizer = Quantizer(build_format(setting.act_format), RangeScaler(symmetric=setting.act_symmetric), Granularity(), batch_axis=0)
        self.integer_mode     = False
        self.float_mode       = False
        
        return None
    
    def forward(self, x: torch.Tensor, x_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* Run the block in the current mode (fake values, integer codes, or float values).
        '''
        
        # Shortcut: the input as it is, or the 1x1 conv into INT8
        identity, identity_params = (x, x_params) if self.shortcut is None else self.shortcut(x, x_params)
        
        # conv1 ends in the activation format, and conv2 stays an INT32 accumulator
        y, y_params = self.conv1(x, x_params)
        y, y_params = self.conv2(y, y_params)
        
        # INT8 residual add, then ReLU and the activation format (the requantization removes the zero point of the add first)
        ops         = mode_ops(self.integer_mode, self.float_mode)
        y, y_params = self.add(y, y_params, identity, identity_params)
        y           = ops.activation(y, y_params, "relu")
        out_params  = self.output_quantizer.params(y) if self.integer_mode else self.output_quantizer.measure(y)
        return ops.requantize(y, y_params, out_params, "mul16_shift"), out_params
