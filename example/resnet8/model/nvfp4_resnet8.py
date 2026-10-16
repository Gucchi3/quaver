# @file nvfp4_resnet8.py
# @brief ResNet-8 with block-scaled convs in the blocks (NVFP4 and the like), with integer math only
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from quaver             import QuantAvgPool2d, QuantConv2d, QuantLinear, QuantParams, build_format
from .nvfp4_basic_block import Nvfp4BasicBlock
from .types             import Resnet8Setting



class Nvfp4Resnet8(nn.Module):
    '''
    #* ResNet-8 with block-scaled convs in the blocks (NVFP4: fp4 weights and ufp4 inputs with a ue4m3 scale for each block of 16), and INT8 or UINT8 codes with static scales outside.
    #? The stem and the classifier have INT8 weights (the input and the output layers). The residuals are UINT8, and ReLU, the adds, and the pool run the integer math on the codes.
    #! The integer mode runs every layer with integers, like QuantResnet8. The layer names are those of Resnet8, so the folded FP32 weights load as they are.
    '''
    
    input_scale: torch.Tensor
    
    def __init__(self, setting: Resnet8Setting) -> None:
        '''
        #* Make the stem, the blocks, the pool, and the classifier.
        #? setting gives the formats of the blocks: the elements of the weights and of the activations, and the format of the block scales.
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
        #* Return the logits of uint8 images in the training mode, and the INT32 codes of the logits with their parameters in the integer mode.
        #? The integer mode gives the codes like the device, so an exported model ends with the classifier. params_values() turns them into the logits.
        '''
        
        # The input is the uint8 code with the scale 1/255: fake values in the training mode, and the codes in the integer mode
        params = QuantParams(self.input_scale, torch.zeros_like(self.input_scale), build_format("uint8"))
        x      = images.long() if self.integer_mode else images.float() * self.input_scale
        
        # The stem with INT8 weights, three blocks with block-scaled convs, the pool on the UINT8 codes, and the classifier with INT8 weights
        x, params = self.stem_conv(x, params)
        x, params = self.stack1(x, params)
        x, params = self.stack2(x, params)
        x, params = self.stack3(x, params)
        x, params = self.pool(x, params)
        x, params = self.classifier(x.flatten(1), params)
        return (x, params) if self.integer_mode else x
