# @file types.py
# @brief 量子化した ResNet-8 の設定
# @author Kohei Yamaguchi, Claude

from dataclasses import dataclass



@dataclass(frozen=True)
class Resnet8Setting:
    '''
    #* 量子化した ResNet-8 1 つの、量子化の設定を持つ。
    #? build.py の MODEL_REGISTRY がモデルの名前ごとに設定を 1 つ与え、QuantResnet8 がそれをもとに層を作る。block_scaled=True のときは、代わりに Nvfp4Resnet8 を作る。
    #? ブロックごとの scale のモデルは、block conv の要素に weight_format と act_format を使い、16 個のブロックごとに scale_format の scale を持つ（NVFP4 なら ue4m3）。
    #! 非対称の量子化（zero point あり）は INT の形式でしか使えないので、FP の形式は対称のままにする。
    '''
    
    weight_format   : str
    act_format      : str
    weight_symmetric: bool = True
    act_symmetric   : bool = True
    block_scaled    : bool = False
    scale_format    : str  = "ue4m3"
