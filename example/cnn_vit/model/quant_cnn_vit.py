# @file quant_cnn_vit.py
# @brief Quantized CNN + ViT made from the quaver samples, with integer math only (no pooling)
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from quaver            import QuantConv2d, QuantLinear, QuantParams, build_format
from .quant_sima_block import QuantSimaBlock
from .types            import CnnVitSetting



class QuantCnnVit(nn.Module):
    '''
    #* Quantized CNN + ViT: UINT8 input (scale 1/255), INT8 weights in the first conv and the classifier, the other layers in the formats of the setting, and an INT8 residual stream.
    #? The layer names are those of CnnVit, so the folded FP32 weights load as they are. switch_integer_mode() moves the whole model to the integer mode. There is no pooling.
    #! It returns real logits in both modes. In the integer mode they come from the exact INT32 accumulator of the classifier.
    '''
    
    input_scale: torch.Tensor
    
    def __init__(self, setting: CnnVitSetting) -> None:
        '''
        #* Make the stem, the blocks, and the classifier.
        #? The first conv and the classifier always have INT8 weights. The last conv of the stem (embed) gives the residual stream (the tokens), without ReLU.
        #? With scale sharing, embed gives its INT32 accumulator and the first block quantizes it to its shared scale (one rounding). Without it, embed gives INT8.
        '''
        
        # Stem: conv1 (INT8 weights) and conv2 with ReLU into the activation format, and embed into the residual (INT32 for the shared scale of the first block, or INT8)
        super().__init__()
        residual   = None if setting.shared_residual else "int8"
        self.conv1 = QuantConv2d(3 , 16, 3, 1, 1, weight_format="int8"               , output_format=setting.act_format, activation="relu", weight_symmetric=setting.weight_symmetric, output_symmetric=setting.act_symmetric)
        self.conv2 = QuantConv2d(16, 32, 3, 2, 1, weight_format=setting.weight_format, output_format=setting.act_format, activation="relu", weight_symmetric=setting.weight_symmetric, output_symmetric=setting.act_symmetric)
        self.embed = QuantConv2d(32, 64, 3, 2, 1, weight_format=setting.weight_format, output_format=residual          , activation="none", weight_symmetric=setting.weight_symmetric, output_symmetric=setting.act_symmetric)
        
        # Two SimA blocks on the 64 tokens, and the classifier on all of them (INT8 weights, INT32 logits)
        self.blocks       = nn.ModuleList([QuantSimaBlock(64, 4, 16, 16, 128, setting) for _ in range(2)])
        self.classifier   = QuantLinear(64 * 8 * 8, 10, weight_format="int8", weight_symmetric=setting.weight_symmetric)
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
        
        # Stem, the SimA blocks, and the classifier on the flattened tokens
        x, params = self.conv1(x, params)
        x, params = self.conv2(x, params)
        x, params = self.embed(x, params)
        for block in self.blocks:
            x, params = block(x, params)
        x, params = self.classifier(x.flatten(1), params)
        return (x, params) if self.integer_mode else x
