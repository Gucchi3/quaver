# @file types.py
# @brief Settings of the quantized CNN + ViT
# @author Kohei Yamaguchi, Claude

from dataclasses import dataclass



@dataclass(frozen=True)
class CnnVitSetting:
    '''
    #* Hold the quantization settings of one quantized CNN + ViT.
    #? MODEL_REGISTRY in build.py gives one setting to each model name, and QuantCnnVit makes its layers from it.
    #! act_format: outputs after ReLU (conv1, conv2, and the MLP). branch_format: inputs of the attention and the MLP (None keeps the INT8 residual as it is).
    #! sima_format: Q, K, the attention map, and V. project_format: the input of the projection (signed: no ReLU6 after attention x V, like the paper). Asymmetric settings work only for INT formats.
    #! shared_residual: one scale for the residual stream inside each block (scale sharing of Kaneko and Meng, Fig. 7). False gives each add its own scale.
    '''
    
    weight_format   : str
    act_format      : str
    branch_format   : str | None
    sima_format     : str
    project_format  : str
    weight_symmetric: bool = True
    act_symmetric   : bool = True
    shared_residual : bool = True
