# @file quant_resnet8.py
# @brief Quantized ResNet-8, made from the quaver samples
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from quaver             import QuantAvgPool2d, QuantConv2d, QuantLinear, QuantParams, build_format
from .quant_basic_block import QuantBasicBlock
from .types             import Resnet8Setting



class QuantResnet8(nn.Module):
    '''
    #* Quantized ResNet-8: UINT8 input (scale 1/255), INT8 stem and classifier weights, block weights and ReLU outputs in the formats of the setting, INT8 residuals, and an average pool on the codes.
    #? It has the layer names of Resnet8, so the folded FP32 weights load as they are. switch_integer_mode() moves the whole model to the integer mode.
    #! It returns real logits in both modes. In the integer mode they come from the exact INT32 accumulator of the classifier.
    '''
    
    input_scale: torch.Tensor
    
    def __init__(self, setting: Resnet8Setting) -> None:
        '''
        #* Make the stem, the blocks, the pool, and the classifier.
        #? The stem and the classifier always have INT8 weights, and their zero points follow the setting like the other layers.
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
        #* Return the logits of uint8 images in the training mode, and the INT32 codes of the logits with their parameters in the integer mode.
        #? The integer mode gives the codes like the device, so an exported model ends with the classifier. params_values() turns them into the logits.
        '''
        
        # The input is the uint8 code with the scale 1/255: fake values in the training mode, and the codes in the integer mode
        params = QuantParams(self.input_scale, torch.zeros_like(self.input_scale), build_format("uint8"))
        x      = images.long() if self.integer_mode else images.float() * self.input_scale
        
        # Stem, three blocks, pool, and classifier
        x, params = self.stem_conv(x, params)
        x, params = self.stack1(x, params)
        x, params = self.stack2(x, params)
        x, params = self.stack3(x, params)
        x, params = self.pool(x, params)
        x, params = self.classifier(x.flatten(1), params)
        return (x, params) if self.integer_mode else x
