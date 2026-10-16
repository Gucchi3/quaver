# @file build.py
# @brief Quantized CNN + ViT models, chosen by name
# @author Kohei Yamaguchi, Claude

##########################################################
'''
Add a model
1. Write a CnnVitSetting: the formats of the weights, of the activations after ReLU, of the inputs of the attention and the MLP, of SimA, and of the projection input.
2. Add "name: setting" to MODEL_REGISTRY.
3. Run main.py --models name.
'''
##########################################################

from .quant_cnn_vit import QuantCnnVit
from .types         import CnnVitSetting

# The first conv and the classifier have INT8 weights, and the residual stream is INT8. 4-bit models put everything else in 4 bits: they requantize the residual to 4 bits
# at the inputs of the attention and the MLP. Symmetric models use unsigned formats after ReLU, and asymmetric models signed formats with zero points (INT only).
# SimA keeps its own symmetric quantizers in every model, and its projection input is signed (no ReLU6 after attention x V, like the paper).
MODEL_REGISTRY: dict[str, CnnVitSetting] = {
    "int8"     : CnnVitSetting(weight_format="int8", act_format="uint8", branch_format=None  , sima_format="int8", project_format="int8", weight_symmetric=True , act_symmetric=True ),
    "int8_asym": CnnVitSetting(weight_format="int8", act_format="int8" , branch_format=None  , sima_format="int8", project_format="int8", weight_symmetric=False, act_symmetric=False),
    "int4"     : CnnVitSetting(weight_format="int4", act_format="uint4", branch_format="int4", sima_format="int4", project_format="int4", weight_symmetric=True , act_symmetric=True ),
    "int4_asym": CnnVitSetting(weight_format="int4", act_format="int4" , branch_format="int4", sima_format="int4", project_format="int4", weight_symmetric=False, act_symmetric=False),
    "fp4"      : CnnVitSetting(weight_format="fp4" , act_format="ufp4" , branch_format="fp4" , sima_format="fp4" , project_format="fp4" , weight_symmetric=True , act_symmetric=True ),
}



def build_model(name: str) -> QuantCnnVit:
    '''
    #* Return the quantized CNN + ViT of the name in MODEL_REGISTRY.
    #? main.py makes each model with this, and then loads the folded FP32 weights into it.
    '''
    
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {', '.join(MODEL_REGISTRY)}")
    return QuantCnnVit(MODEL_REGISTRY[name])
