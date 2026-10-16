# @file build.py
# @brief Quantized ResNet-8 models, chosen by name
# @author Kohei Yamaguchi, Claude

##########################################################
'''
Add a model
1. Write a Resnet8Setting: the formats of the weights and of the activations, whether each is symmetric, and whether the scales are for blocks (NVFP4).
2. Add "name: setting" to MODEL_REGISTRY.
3. Run main.py --models name.
'''
##########################################################

from .nvfp4_resnet8 import Nvfp4Resnet8
from .quant_resnet8 import QuantResnet8
from .types         import Resnet8Setting

# Symmetric models keep every zero point at 0, so they use unsigned formats after ReLU.
# Asymmetric models give every weight and activation a zero point, so they use signed formats after ReLU (0 goes to the lowest code). FP formats are only symmetric.
# nvfp4 puts NVFP4 on the weights of the block convs and block-scaled UFP4 on their inputs (blocks of 16 with UE4M3 scales). The stem, the classifier, and the residuals stay INT8 / UINT8.
MODEL_REGISTRY: dict[str, Resnet8Setting] = {
    "int8"     : Resnet8Setting(weight_format="int8", act_format="uint8", weight_symmetric=True , act_symmetric=True , block_scaled=False),
    "int8_asym": Resnet8Setting(weight_format="int8", act_format="int8" , weight_symmetric=False, act_symmetric=False, block_scaled=False),
    "int4"     : Resnet8Setting(weight_format="int4", act_format="uint4", weight_symmetric=True , act_symmetric=True , block_scaled=False),
    "int4_asym": Resnet8Setting(weight_format="int4", act_format="int4" , weight_symmetric=False, act_symmetric=False, block_scaled=False),
    "fp4"      : Resnet8Setting(weight_format="fp4" , act_format="ufp4" , weight_symmetric=True , act_symmetric=True , block_scaled=False),
    "nvfp4"    : Resnet8Setting(weight_format="fp4" , act_format="ufp4" , weight_symmetric=True , act_symmetric=True , block_scaled=True ),
}



def build_model(name: str) -> QuantResnet8 | Nvfp4Resnet8:
    '''
    #* Return the quantized ResNet-8 of the name in MODEL_REGISTRY: Nvfp4Resnet8 for a block-scaled setting, and QuantResnet8 for the others.
    #? main.py makes each model with this, and then loads the folded FP32 weights into it.
    '''
    
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {', '.join(MODEL_REGISTRY)}")
    setting = MODEL_REGISTRY[name]
    return Nvfp4Resnet8(setting) if setting.block_scaled else QuantResnet8(setting)
