# @file types.py
# @brief Settings of the quantized ResNet-8
# @author Kohei Yamaguchi, Claude

from dataclasses import dataclass



@dataclass(frozen=True)
class Resnet8Setting:
    '''
    #* Hold the quantization settings of one quantized ResNet-8.
    #? MODEL_REGISTRY in build.py gives one setting to each model name, and QuantResnet8 makes its layers from it. block_scaled=True makes Nvfp4Resnet8 instead.
    #? Block-scaled models use weight_format and act_format for the elements of the block convs, with a scale for each block of 16 in scale_format (ue4m3 for NVFP4).
    #! Asymmetric quantization (with zero points) works only for INT formats, so FP formats stay symmetric.
    '''
    
    weight_format   : str
    act_format      : str
    weight_symmetric: bool = True
    act_symmetric   : bool = True
    block_scaled    : bool = False
    scale_format    : str  = "ue4m3"
